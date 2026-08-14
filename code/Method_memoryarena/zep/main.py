import argparse
import asyncio
import copy
import contextlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv
from openai import AsyncOpenAI
from neo4j import AsyncGraphDatabase

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_CROSS_ENCODER

from .token_tracker import TokenTracker
from ..dataset_hygiene import natural_session_keys, resolve_required_endpoint


DEFAULT_LLM_API_KEY = "empty"
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_LLM_MODEL = "Qwen3.5-9B"
DEFAULT_EMBEDDING_API_KEY = "empty"
DEFAULT_EMBEDDING_BASE_URL = None
DEFAULT_EMBEDDING_MODEL = "/path/to/local/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_EMBEDDING_MAX_INPUT_TOKENS = int(os.getenv("ZEP_EMBEDDING_MAX_INPUT_TOKENS", "240"))
DEFAULT_EMBEDDING_TOKENIZER_PATH = os.getenv(
    "ZEP_EMBEDDING_TOKENIZER_PATH", "/path/to/local/all-MiniLM-L6-v2"
)
_EMBEDDING_TOKENIZER = None
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "neo4jneo4j"
DEFAULT_ANSWER_TEMPERATURE = 0.0
DEFAULT_ANSWER_MAX_TOKENS = 200
QWEN_DISABLE_THINKING_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
DEFAULT_LLM_CONTEXT_WINDOW = int(os.getenv("ZEP_LLM_CONTEXT_WINDOW", "20000"))
DEFAULT_LLM_SAFETY_MARGIN = int(os.getenv("ZEP_LLM_SAFETY_MARGIN", "256"))
DEFAULT_LLM_MAX_COMPLETION_TOKENS = int(os.getenv("ZEP_LLM_MAX_COMPLETION_TOKENS", "4096"))
DEFAULT_QWEN_TOKENIZER_PATH = os.getenv("ZEP_QWEN_TOKENIZER_PATH", "/path/to/local/Qwen3.5-9B")
_TOKENIZER = None
_TOKENIZER_LOAD_FAILED = False


def get_embedding_tokenizer():
    global _EMBEDDING_TOKENIZER
    if _EMBEDDING_TOKENIZER is None:
        from transformers import AutoTokenizer

        _EMBEDDING_TOKENIZER = AutoTokenizer.from_pretrained(
            DEFAULT_EMBEDDING_TOKENIZER_PATH,
            local_files_only=True,
        )
    return _EMBEDDING_TOKENIZER


def truncate_embedding_text(text):
    if not isinstance(text, str):
        return text
    tokenizer = get_embedding_tokenizer()
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= DEFAULT_EMBEDDING_MAX_INPUT_TOKENS:
        return text
    truncated = tokenizer.decode(
        token_ids[:DEFAULT_EMBEDDING_MAX_INPUT_TOKENS],
        skip_special_tokens=True,
    )
    logger.warning(
        "Truncated embedding input from %d to %d tokens for the 256-token endpoint.",
        len(token_ids),
        DEFAULT_EMBEDDING_MAX_INPUT_TOKENS,
    )
    return truncated


class TruncatingOpenAIEmbedder(OpenAIEmbedder):
    async def create(self, input_data):
        if isinstance(input_data, list) and input_data and all(
            isinstance(item, str) for item in input_data
        ):
            input_data = [truncate_embedding_text(item) for item in input_data]
        else:
            input_data = truncate_embedding_text(input_data)
        return await super().create(input_data)

    async def create_batch(self, input_data_list):
        return await super().create_batch(
            [truncate_embedding_text(item) for item in input_data_list]
        )


def get_qwen_tokenizer():
    global _TOKENIZER, _TOKENIZER_LOAD_FAILED
    if _TOKENIZER is not None:
        return _TOKENIZER
    if _TOKENIZER_LOAD_FAILED:
        return None
    try:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(
            DEFAULT_QWEN_TOKENIZER_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        return _TOKENIZER
    except Exception as e:
        _TOKENIZER_LOAD_FAILED = True
        logger.warning(
            "Could not load tokenizer from %s; falling back to conservative char-based truncation: %s",
            DEFAULT_QWEN_TOKENIZER_PATH,
            e,
        )
        return None


def get_message_role(message):
    if isinstance(message, dict):
        return str(message.get("role", "user"))
    return str(getattr(message, "role", "user"))


def get_message_content(message):
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def normalize_content_for_counting(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except TypeError:
        return str(content)


def clone_message_with_content(message, content):
    if isinstance(message, dict):
        updated = dict(message)
        updated["content"] = content
        return updated
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": content})
    updated = type("MessageProxy", (), {})()
    updated.__dict__.update(getattr(message, "__dict__", {}))
    updated.content = content
    return updated


def normalize_messages_for_tokenizer(messages):
    return [
        {
            "role": get_message_role(message),
            "content": normalize_content_for_counting(get_message_content(message)),
        }
        for message in messages
    ]


def count_text_tokens(text):
    tokenizer = get_qwen_tokenizer()
    if tokenizer is None:
        return len(text)
    return len(tokenizer.encode(text, add_special_tokens=False))


def count_chat_tokens(messages, extra_body=None):
    normalized_messages = normalize_messages_for_tokenizer(messages)
    tokenizer = get_qwen_tokenizer()
    if tokenizer is None:
        return sum(len(message["content"]) + 4 for message in normalized_messages) + 8

    chat_template_kwargs = dict((extra_body or {}).get("chat_template_kwargs") or {})
    try:
        return len(
            tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=True,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
        )
    except TypeError:
        try:
            return len(
                tokenizer.apply_chat_template(
                    normalized_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            pass
    except Exception:
        pass

    return sum(
        len(tokenizer.encode(message["content"], add_special_tokens=False)) + 4
        for message in normalized_messages
    ) + 8


def truncate_text_to_tokens(text, token_limit):
    if token_limit <= 0:
        return ""
    tokenizer = get_qwen_tokenizer()
    marker = f"\n...[truncated to fit {DEFAULT_LLM_CONTEXT_WINDOW}-token context window]...\n"
    if tokenizer is None:
        if len(text) <= token_limit:
            return text
        marker_chars = len(marker)
        available = max(1, token_limit - marker_chars)
        head_chars = max(1, available // 2)
        tail_chars = max(1, available - head_chars)
        return text[:head_chars] + marker + text[-tail_chars:]

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= token_limit:
        return text
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    available = max(1, token_limit - len(marker_ids))
    head_tokens = max(1, available // 2)
    tail_tokens = max(1, available - head_tokens)
    return (
        tokenizer.decode(token_ids[:head_tokens], skip_special_tokens=False)
        + marker
        + tokenizer.decode(token_ids[-tail_tokens:], skip_special_tokens=False)
    )


def truncate_messages_to_budget(messages, input_token_budget, extra_body=None):
    prepared_messages = list(messages)
    before_tokens = count_chat_tokens(prepared_messages, extra_body)
    current_tokens = before_tokens
    if current_tokens <= input_token_budget:
        return prepared_messages, before_tokens, current_tokens, False

    for _ in range(12):
        candidates = []
        for idx, message in enumerate(prepared_messages):
            content = get_message_content(message)
            if not isinstance(content, str) or not content:
                continue
            role = get_message_role(message).lower()
            token_count = count_text_tokens(content)
            candidates.append((role == "system", token_count, idx, content))

        if not candidates:
            break

        non_system = [candidate for candidate in candidates if not candidate[0]]
        _, content_tokens, idx, content = max(non_system or candidates, key=lambda item: item[1])
        overflow = max(1, current_tokens - input_token_budget)
        target_content_tokens = max(32, content_tokens - overflow - DEFAULT_LLM_SAFETY_MARGIN)
        truncated_content = truncate_text_to_tokens(content, target_content_tokens)
        if truncated_content == content:
            break

        prepared_messages[idx] = clone_message_with_content(prepared_messages[idx], truncated_content)
        current_tokens = count_chat_tokens(prepared_messages, extra_body)
        if current_tokens <= input_token_budget:
            break

    return prepared_messages, before_tokens, current_tokens, True


def coerce_positive_int(value, default):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def add_json_schema_instruction(messages, response_format):
    schema = (response_format.get("json_schema") or {}).get("schema") or {}
    if not schema:
        return list(messages)
    instruction = (
        "\n\n# RESPONSE FORMAT\n"
        "Return exactly one JSON object matching the JSON Schema below. "
        "Use the exact property names. Do not return a top-level array, prose, or markdown.\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    )
    prepared_messages = list(messages)
    for index in range(len(prepared_messages) - 1, -1, -1):
        message = prepared_messages[index]
        content = get_message_content(message)
        if get_message_role(message).lower() == "user" and isinstance(content, str):
            prepared_messages[index] = clone_message_with_content(message, content + instruction)
            break
    return prepared_messages


def prepare_chat_completion_kwargs(kwargs, default_extra_body=None):
    prepared = dict(kwargs)
    existing = prepared.get("extra_body") or {}
    merged = dict(default_extra_body or {})
    merged.update(existing)
    default_chat_template_kwargs = dict(
        (default_extra_body or {}).get("chat_template_kwargs") or {}
    )
    existing_chat_template_kwargs = dict(existing.get("chat_template_kwargs") or {})
    if default_chat_template_kwargs or existing_chat_template_kwargs:
        default_chat_template_kwargs.update(existing_chat_template_kwargs)
        merged["chat_template_kwargs"] = default_chat_template_kwargs
    if merged:
        prepared["extra_body"] = merged

    messages = prepared.get("messages")
    if not messages:
        return prepared

    response_format = prepared.get("response_format") or {}
    if (
        "deepseek" in str(prepared.get("model", "")).lower()
        and response_format.get("type") == "json_schema"
    ):
        messages = add_json_schema_instruction(messages, response_format)
        prepared["messages"] = messages

    if "max_tokens" in prepared:
        max_token_key = "max_tokens"
    elif "max_completion_tokens" in prepared:
        max_token_key = "max_completion_tokens"
    else:
        max_token_key = "max_tokens"
    requested_completion_tokens = coerce_positive_int(
        prepared.get(max_token_key),
        DEFAULT_LLM_MAX_COMPLETION_TOKENS,
    )
    completion_tokens = min(requested_completion_tokens, DEFAULT_LLM_MAX_COMPLETION_TOKENS)
    if completion_tokens != requested_completion_tokens:
        logger.warning(
            "Reducing requested completion tokens from %s to %s for %s-token context window.",
            requested_completion_tokens,
            completion_tokens,
            DEFAULT_LLM_CONTEXT_WINDOW,
        )
    prepared[max_token_key] = completion_tokens

    input_token_budget = DEFAULT_LLM_CONTEXT_WINDOW - completion_tokens - DEFAULT_LLM_SAFETY_MARGIN
    if input_token_budget < 128:
        completion_tokens = max(1, DEFAULT_LLM_CONTEXT_WINDOW - DEFAULT_LLM_SAFETY_MARGIN - 128)
        prepared[max_token_key] = completion_tokens
        input_token_budget = DEFAULT_LLM_CONTEXT_WINDOW - completion_tokens - DEFAULT_LLM_SAFETY_MARGIN

    truncated_messages, before_tokens, after_tokens, was_truncated = truncate_messages_to_budget(
        messages,
        input_token_budget,
        prepared.get("extra_body"),
    )
    prepared["messages"] = truncated_messages
    if was_truncated:
        logger.warning(
            "Truncated chat input from %s to %s tokens; input budget=%s, completion budget=%s, context window=%s.",
            before_tokens,
            after_tokens,
            input_token_budget,
            completion_tokens,
            DEFAULT_LLM_CONTEXT_WINDOW,
        )

    return prepared


def parse_json_value(value):
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def resolve_json_schema(schema, root_schema):
    reference = schema.get("$ref") if isinstance(schema, dict) else None
    if not reference or not reference.startswith("#/$defs/"):
        return schema
    definition_name = reference.rsplit("/", 1)[-1]
    return (root_schema.get("$defs") or {}).get(definition_name, schema)


def normalize_json_to_schema(value, schema, root_schema):
    schema = resolve_json_schema(schema, root_schema)
    schema_type = schema.get("type") if isinstance(schema, dict) else None
    if schema_type == "array" and isinstance(value, list):
        item_schema = schema.get("items") or {}
        return [normalize_json_to_schema(item, item_schema, root_schema) for item in value]
    if schema_type != "object" or not isinstance(value, dict):
        return value

    properties = schema.get("properties") or {}
    normalized = dict(value)
    array_property_names = [
        name
        for name, property_schema in properties.items()
        if resolve_json_schema(property_schema, root_schema).get("type") == "array"
    ]
    for property_name, property_schema in properties.items():
        if property_name not in normalized and property_name == "name":
            if "entity_name" in normalized:
                normalized[property_name] = normalized["entity_name"]
            else:
                aliases = [key for key in normalized if key.endswith("_name")]
                if len(aliases) == 1:
                    normalized[property_name] = normalized[aliases[0]]
        if (
            property_name not in normalized
            and len(array_property_names) == 1
            and property_name == array_property_names[0]
        ):
            aliases = [
                key
                for key, candidate in normalized.items()
                if key not in properties and isinstance(candidate, list)
            ]
            if len(aliases) == 1:
                normalized[property_name] = normalized[aliases[0]]
        if property_name in normalized:
            normalized[property_name] = normalize_json_to_schema(
                normalized[property_name], property_schema, root_schema
            )
    return normalized


def normalize_structured_response(response, request_kwargs):
    response_format = request_kwargs.get("response_format") or {}
    if response_format.get("type") not in {"json_schema", "json_object"} or not response.choices:
        return response

    message = response.choices[0].message
    root_schema = (response_format.get("json_schema") or {}).get("schema") or {}
    sources = [
        ("content", message.content),
        ("reasoning_content", getattr(message, "reasoning_content", None)),
    ]
    parsed = None
    source_name = None
    for candidate_source, candidate in sources:
        parsed = parse_json_value(candidate)
        if isinstance(parsed, (dict, list)):
            source_name = candidate_source
            break

    normalized = parsed
    wrapped_property = None
    if isinstance(parsed, list):
        schema = root_schema
        properties = schema.get("properties") or {}
        array_properties = [
            name
            for name, property_schema in properties.items()
            if property_schema.get("type") == "array"
        ]
        if len(array_properties) != 1:
            return response
        wrapped_property = array_properties[0]
        normalized = {wrapped_property: parsed}

    if not isinstance(normalized, dict):
        return response

    normalized = normalize_json_to_schema(normalized, root_schema, root_schema)
    normalized_content = json.dumps(normalized, ensure_ascii=False)
    if normalized_content != message.content:
        message.content = normalized_content
        logger.warning(
            "Normalized structured response from %s%s.",
            source_name,
            f" by wrapping list in '{wrapped_property}'" if wrapped_property else "",
        )
    return response


def structured_response_is_valid(response, request_kwargs):
    if not response.choices:
        return False
    content = response.choices[0].message.content
    response_format = request_kwargs.get("response_format") or {}
    if response_format.get("type") not in {"json_schema", "json_object"}:
        return isinstance(content, str) and bool(content.strip())
    return isinstance(parse_json_value(content), dict)


def is_retryable_llm_error(exc):
    status_code = getattr(exc, "status_code", None)
    if status_code in {403, 408, 409, 429}:
        return True
    if isinstance(status_code, int) and status_code >= 500:
        return True
    error_name = type(exc).__name__.lower()
    error_text = str(exc).lower()
    return any(
        marker in error_name or marker in error_text
        for marker in (
            "connection",
            "timeout",
            "temporarily unavailable",
            "cloudflare",
            "just a moment",
        )
    )


class ExtraBodyCompletions:
    def __init__(self, completions, extra_body):
        self._completions = completions
        self._extra_body = extra_body
        configured_concurrency = os.getenv("ZEP_LLM_CONCURRENCY")
        self._semaphore = (
            asyncio.Semaphore(coerce_positive_int(configured_concurrency, 1))
            if configured_concurrency
            else None
        )
        try:
            self._response_retries = max(
                0, int(os.getenv("ZEP_LLM_RESPONSE_RETRIES", "2"))
            )
        except ValueError:
            self._response_retries = 2

    async def create(self, *args, **kwargs):
        prepared = prepare_chat_completion_kwargs(kwargs, self._extra_body)
        response = None
        for attempt in range(self._response_retries + 1):
            try:
                if self._semaphore is None:
                    response = await self._completions.create(*args, **prepared)
                else:
                    async with self._semaphore:
                        response = await self._completions.create(*args, **prepared)
            except Exception as exc:
                if attempt >= self._response_retries or not is_retryable_llm_error(exc):
                    raise
                delay = min(2 ** attempt, 30)
                logger.warning(
                    "Retrying transient LLM request error (%s/%s) in %ss: %s",
                    attempt + 1,
                    self._response_retries,
                    delay,
                    str(exc)[:300],
                )
                await asyncio.sleep(delay)
                continue
            response = normalize_structured_response(response, prepared)
            if structured_response_is_valid(response, prepared):
                return response
            if attempt < self._response_retries:
                logger.warning(
                    "Retrying empty or malformed structured response (%s/%s).",
                    attempt + 1,
                    self._response_retries,
                )
                await asyncio.sleep(min(2 ** attempt, 4))
        return response

    def __getattr__(self, name):
        return getattr(self._completions, name)


class ExtraBodyChat:
    def __init__(self, chat, extra_body):
        self._chat = chat
        self.completions = ExtraBodyCompletions(chat.completions, extra_body)

    def __getattr__(self, name):
        return getattr(self._chat, name)


class ExtraBodyAsyncOpenAI:
    def __init__(self, client, extra_body=None):
        self._client = client
        self.chat = ExtraBodyChat(client.chat, extra_body)

    def __getattr__(self, name):
        return getattr(self._client, name)


def build_openai_client(api_key, base_url, model):
    extra_body = (
        QWEN_DISABLE_THINKING_EXTRA_BODY
        if "qwen" in str(model).lower()
        else None
    )
    return ExtraBodyAsyncOpenAI(
        AsyncOpenAI(api_key=api_key, base_url=base_url),
        extra_body=extra_body,
    )


logging.basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


ANSWER_SYSTEM_PROMPT = """You are a helpful expert assistant answering user questions from retrieved conversation memories.
Answer briefly and precisely using only the provided context. If the context is insufficient, abstain.

When interpreting memories, use the timestamp to determine when an event happened, not when someone talked about it.

Example:
Memory: (2023-03-15T16:33:00Z) I went to the vet yesterday.
Question: What day did I go to the vet?
Correct answer: March 15, 2023
"""


def ensure_parent_dir(path):
    if path:
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)


def parse_datetime_string(datetime_str):
    """Parse a LOCOMO datetime string like '1:56 pm on 8 May, 2023'."""
    pattern = r"(\d{1,2}):(\d{2})\s+(am|pm)\s+on\s+(\d{1,2})\s+(\w+),\s+(\d{4})"
    match = re.match(pattern, datetime_str)

    if not match:
        raise ValueError(f"Cannot parse datetime string: {datetime_str}")

    hour, minute, ampm, day, month_name, year = match.groups()

    hour = int(hour)
    if ampm.lower() == "pm" and hour != 12:
        hour += 12
    elif ampm.lower() == "am" and hour == 12:
        hour = 0

    month_map = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }

    month = month_map.get(month_name)
    if not month:
        raise ValueError(f"Unknown month: {month_name}")

    return datetime(int(year), month, int(day), hour, int(minute), tzinfo=timezone.utc)


def format_dialogue_episode(speaker, text, blip_caption=None):
    """Format a dialogue message into a Graphiti episode body."""
    episode_body = f"\n{speaker}: {text}"
    if blip_caption:
        episode_body += f" (image caption: {blip_caption})"
    return episode_body


def extract_retrieved_facts(results):
    retrieved_facts = {"Edges": [], "Nodes": [], "Episodes": [], "Communities": []}

    try:
        for edge in results.edges[:5]:
            fact = f"{edge.name}: {edge.fact}"
            time_info = []
            if hasattr(edge, "expired_at") and edge.expired_at:
                time_info.append(f" (Expired at: {edge.expired_at})")
            if hasattr(edge, "valid_at") and edge.valid_at:
                time_info.append(f" (Valid from: {edge.valid_at})")
            if hasattr(edge, "invalid_at") and edge.invalid_at:
                time_info.append(f" (Valid until: {edge.invalid_at})")
            if time_info:
                fact += "".join(time_info)
            retrieved_facts["Edges"].append(fact)
    except Exception:
        pass

    try:
        for node in results.nodes[:5]:
            retrieved_facts["Nodes"].append(f"{node.name}: {node.summary}")
    except Exception:
        pass

    try:
        for episode in results.episodes:
            retrieved_facts["Episodes"].append(
                f"{episode.source_description}: {episode.content}"
            )
    except Exception:
        pass

    try:
        for community in results.communities[:3]:
            retrieved_facts["Communities"].append(f"{community.name}: {community.summary}")
    except Exception:
        pass

    return retrieved_facts


def build_answer_context(retrieved):
    sections = []
    for section_name in ("Edges", "Nodes", "Episodes", "Communities"):
        values = retrieved.get(section_name, [])
        if values:
            section_text = "\n".join(values)
            sections.append(f"[{section_name}]\n{section_text}")
    return "\n\n".join(sections)


def sample_has_completed_responses(sample_result, expected_qa_count=None):
    qa_items = sample_result.get("qa", [])
    if not qa_items:
        return False
    if expected_qa_count is not None and len(qa_items) != expected_qa_count:
        return False
    return all(item.get("response") for item in qa_items)


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_api_key(value):
    if not isinstance(value, str) or not value.startswith("env:"):
        return value
    variable_name = value.split(":", 1)[1].strip()
    if not variable_name:
        raise ValueError("API key environment variable name is empty")
    resolved = os.getenv(variable_name)
    if not resolved:
        raise ValueError(f"Required API key environment variable is not set: {variable_name}")
    return resolved


def get_summary_path(output_path):
    base, ext = os.path.splitext(output_path)
    return f"{base}_summary{ext or '.json'}"


def collect_retrieval_latencies(results):
    latencies = []
    for sample in results:
        for qa_item in sample.get("qa", []):
            latency = qa_item.get("retrieval_latency_seconds")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))
    return latencies


def build_run_summary(results):
    latencies = collect_retrieval_latencies(results)
    total_questions = sum(len(sample.get("qa", [])) for sample in results)
    average = sum(latencies) / len(latencies) if latencies else None
    return {
        "total_samples": len(results),
        "total_questions": total_questions,
        "retrieval_latency_count": len(latencies),
        "average_retrieval_latency_seconds": average,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def atomic_json_dump(path, value):
    ensure_parent_dir(path)
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)


def save_results_and_summary(output_path, results):
    atomic_json_dump(output_path, results)
    summary = build_run_summary(results)
    summary_path = get_summary_path(output_path)
    atomic_json_dump(summary_path, summary)
    return summary, summary_path


async def clear_neo4j_database(uri, user, password):
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
    finally:
        await driver.close()


async def clear_neo4j_group(uri, user, password, group_id):
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        async with driver.session() as session:
            await session.run(
                "MATCH (n {group_id: $group_id}) DETACH DELETE n",
                group_id=group_id,
            )
            await session.run(
                "MATCH ()-[r]->() WHERE r.group_id = $group_id DELETE r",
                group_id=group_id,
            )
    finally:
        await driver.close()


async def generate_answer(
    question,
    retrieved,
    answer_client,
    answer_model,
    answer_temperature=DEFAULT_ANSWER_TEMPERATURE,
    answer_max_tokens=DEFAULT_ANSWER_MAX_TOKENS,
):
    context = build_answer_context(retrieved)
    if not context.strip():
        return "Insufficient context to answer."

    try:
        response = await answer_client.chat.completions.create(
            model=answer_model,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "# CONTEXT\n"
                        f"{context}\n\n"
                        "# QUESTION\n"
                        f"{question}\n\n"
                        "Answer briefly and directly based only on the context."
                    ),
                },
            ],
            max_tokens=answer_max_tokens,
            temperature=answer_temperature,
        )
        return response.choices[0].message.content.strip() if response.choices[0].message.content else "No response generated"
    except Exception as e:
        print(f"Error generating answer for question '{question}': {e}")
        return "Error: Unable to generate answer"


async def run_zep(
    dataset_path,
    output_path,
    token_file=None,
    llm_model=DEFAULT_LLM_MODEL,
    llm_small_model=None,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL,
    embedding_api_key=DEFAULT_EMBEDDING_API_KEY,
    embedding_base_url=DEFAULT_EMBEDDING_BASE_URL,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    neo4j_uri=DEFAULT_NEO4J_URI,
    neo4j_user=DEFAULT_NEO4J_USER,
    neo4j_password=DEFAULT_NEO4J_PASSWORD,
    answer_model=None,
    answer_api_key=None,
    answer_base_url=None,
    answer_temperature=DEFAULT_ANSWER_TEMPERATURE,
    answer_max_tokens=DEFAULT_ANSWER_MAX_TOKENS,
    clear_neo4j=False,
    sample_concurrency=1,
    track_tokens=False,
    retrieval_top_k=10,
):
    load_dotenv()

    llm_model = llm_model or DEFAULT_LLM_MODEL
    llm_small_model = llm_small_model or llm_model
    llm_api_key = resolve_api_key(llm_api_key or DEFAULT_LLM_API_KEY)
    llm_base_url = llm_base_url or DEFAULT_LLM_BASE_URL
    embedding_model_name = embedding_model_name or DEFAULT_EMBEDDING_MODEL
    embedding_api_key = resolve_api_key(embedding_api_key or DEFAULT_EMBEDDING_API_KEY)
    embedding_base_url = resolve_required_endpoint(embedding_base_url)
    embedding_dim = embedding_dim or DEFAULT_EMBEDDING_DIM
    neo4j_uri = neo4j_uri or DEFAULT_NEO4J_URI
    neo4j_user = neo4j_user or DEFAULT_NEO4J_USER
    neo4j_password = neo4j_password or DEFAULT_NEO4J_PASSWORD
    answer_model = answer_model or llm_model
    answer_api_key = resolve_api_key(answer_api_key or llm_api_key)
    sample_concurrency = coerce_positive_int(sample_concurrency, 1)
    answer_base_url = answer_base_url or llm_base_url
    token_file = token_file or output_path.replace(".json", "_tokens.json")
    retrieval_top_k = int(retrieval_top_k or 10)
    if retrieval_top_k <= 0:
        raise ValueError("retrieval_top_k must be positive")

    ensure_parent_dir(output_path)
    ensure_parent_dir(token_file)

    tracker = TokenTracker(output_file=token_file) if coerce_bool(track_tokens) else None

    def tracker_stage(name):
        return tracker.stage(name) if tracker is not None else contextlib.nullcontext()

    if coerce_bool(clear_neo4j):
        print(f"Clearing Neo4j database at {neo4j_uri} before importing LOCOMO data...")
        await clear_neo4j_database(neo4j_uri, neo4j_user, neo4j_password)
        print("Neo4j database cleared successfully.")

    llm_config = LLMConfig(
        api_key=llm_api_key,
        model=llm_model,
        small_model=llm_small_model,
        base_url=llm_base_url,
    )
    llm_client = OpenAIGenericClient(
        config=llm_config,
        client=build_openai_client(llm_api_key, llm_base_url, llm_model),
    )
    answer_client = build_openai_client(answer_api_key, answer_base_url, answer_model)

    graphiti = None
    try:
        graphiti = Graphiti(
            neo4j_uri,
            neo4j_user,
            neo4j_password,
            llm_client=llm_client,
            embedder=TruncatingOpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    embedding_model=embedding_model_name,
                    api_key=embedding_api_key,
                    base_url=embedding_base_url,
                    embedding_dim=embedding_dim,
                )
            ),
            cross_encoder=OpenAIRerankerClient(
                config=llm_config,
                client=build_openai_client(llm_api_key, llm_base_url, llm_model),
            ),
        )

        search_config = copy.deepcopy(COMBINED_HYBRID_SEARCH_CROSS_ENCODER)
        search_config.limit = retrieval_top_k
        print(f"Using retrieval top-k: {retrieval_top_k}")
        print(f"Using sample concurrency: {sample_concurrency}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            locomo_samples = json.load(f)

        expected_qa_counts = {
            sample.get("sample_id", f"sample_{idx}"): sum(
                1 for qa in sample.get("qa", []) if "question" in qa
            )
            for idx, sample in enumerate(locomo_samples)
        }
        all_results = []
        results_by_sample_id = {}
        processed_sample_ids = set()
        resume_from_output = os.path.exists(output_path) and not coerce_bool(clear_neo4j)
        if os.path.exists(output_path) and coerce_bool(clear_neo4j):
            print(f"Ignoring existing results at {output_path} because clear_neo4j is enabled.")
        if resume_from_output:
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                for result in existing_results:
                    sample_id = result["sample_id"]
                    all_results.append(result)
                    results_by_sample_id[sample_id] = result
                    if sample_has_completed_responses(
                        result, expected_qa_counts.get(sample_id)
                    ):
                        processed_sample_ids.add(sample_id)
                    else:
                        print(
                            f"Resuming incomplete sample {sample_id}: "
                            f"{len(result.get('qa', []))}/{expected_qa_counts.get(sample_id, 0)} queries saved"
                        )
                print(
                    f"Loaded {len(processed_sample_ids)} completed and "
                    f"{len(all_results) - len(processed_sample_ids)} incomplete samples"
                )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load existing results from {output_path}: {e}")

        result_order = {
            sample.get("sample_id", f"sample_{idx}"): idx
            for idx, sample in enumerate(locomo_samples)
        }
        checkpoint_lock = asyncio.Lock()
        sample_semaphore = asyncio.Semaphore(sample_concurrency)

        async def save_checkpoint():
            async with checkpoint_lock:
                all_results.sort(
                    key=lambda item: result_order.get(item.get("sample_id"), len(result_order))
                )
                return save_results_and_summary(output_path, all_results)

        async def process_sample(sample_idx, locomo_data):
            async with sample_semaphore:
                namespace = locomo_data.get("sample_id", f"sample_{sample_idx}")
                if namespace in processed_sample_ids:
                    print(
                        f"Skipping sample {sample_idx + 1}/{len(locomo_samples)} "
                        f"(ID: {namespace}) - already processed"
                    )
                    return
    
                print(f"\n=== Processing sample {sample_idx + 1}/{len(locomo_samples)} (ID: {namespace}) ===")
                is_resuming_sample = namespace in results_by_sample_id
                sample_result = results_by_sample_id.get(namespace)
                if sample_result is None:
                    sample_result = {
                        "sample_id": namespace,
                        "qa": [],
                        "_zep_indexing_completed": False,
                    }
                    all_results.append(sample_result)
                    results_by_sample_id[namespace] = sample_result
                    await save_checkpoint()
    
                if not sample_result.get("_zep_indexing_completed", False):
                    if is_resuming_sample:
                        print(f"Clearing partial Neo4j data for incomplete sample {namespace}...")
                        await clear_neo4j_group(
                            neo4j_uri, neo4j_user, neo4j_password, namespace
                        )
                    conversation = locomo_data["conversation"]
                    session_keys = natural_session_keys(conversation)
    
                    with tracker_stage(f"Sample {namespace}"):
                        for session_key in session_keys:
                            session_num = session_key.split("_")[1]
                            datetime_key = f"session_{session_num}_date_time"
                            if datetime_key not in conversation:
                                continue
    
                            datetime_str = conversation[datetime_key]
                            reference_time = parse_datetime_string(datetime_str)
                            session_dialogues = conversation[session_key]
    
                            with tracker_stage(f"Session {session_key}"):
                                for dialogue_idx, dialogue in enumerate(session_dialogues):
                                    with tracker_stage(f"Dialog {dialogue_idx}"):
                                        episode_body = format_dialogue_episode(
                                            dialogue["speaker"],
                                            dialogue["text"],
                                            dialogue.get("blip_caption"),
                                        )
                                        try:
                                            await graphiti.add_episode(
                                                name=f"Conversation Session {session_num} - Dialogue {dialogue_idx}",
                                                episode_body=episode_body,
                                                source=EpisodeType.message,
                                                source_description=(
                                                    f"conversation between {conversation.get('speaker_a', 'Speaker A')} "
                                                    f"and {conversation.get('speaker_b', 'Speaker B')} on {datetime_str}"
                                                ),
                                                reference_time=reference_time,
                                                group_id=namespace,
                                            )
                                        except Exception as e:
                                            raise RuntimeError(
                                                f"Failed to add episode for sample {namespace}, "
                                                f"session {session_num}, dialogue {dialogue_idx}"
                                            ) from e
    
                    print(f"Finished processing sample {namespace}. Total sessions processed: {len(session_keys)}")
                    print(f"Building communities for sample {namespace}...")
                    await graphiti.build_communities(group_ids=[namespace])
                    print(f"Communities built successfully for sample {namespace}.")
                    sample_result["_zep_indexing_completed"] = True
                    await save_checkpoint()
                    print(f"Saved indexing checkpoint for sample {namespace}.")
                else:
                    print(f"Reusing completed Neo4j index for sample {namespace}.")
    
                qa_list = [qa for qa in locomo_data.get("qa", []) if "question" in qa]
                print(f"\nSearching {len(qa_list)} questions for sample {namespace}:")
                completed_qa_count = len(sample_result.get("qa", []))
                if completed_qa_count:
                    print(
                        f"Resuming query retrieval at {completed_qa_count + 1}/{len(qa_list)} "
                        f"for sample {namespace}."
                    )
    
                for qa_idx, qa_item in enumerate(qa_list):
                    if qa_idx < completed_qa_count:
                        continue
    
                    question = qa_item["question"]
                    expected_answer = qa_item.get("answer", "N/A")
                    category = qa_item.get("category")
    
                    print(f"\n--- Question {qa_idx + 1}/{len(qa_list)} ---")
                    print(f"Question: {question}")
                    print(f"Expected Answer: {expected_answer}")
    
                    retrieval_latency_seconds = None
                    try:
                        retrieval_started = time.perf_counter()
                        results = await graphiti.search_(question, config=search_config, group_ids=[namespace])
                        retrieved_facts = extract_retrieved_facts(results)
                        retrieval_latency_seconds = time.perf_counter() - retrieval_started
                        print(f"Retrieval latency: {retrieval_latency_seconds:.3f}s")
    
                        response = await generate_answer(
                            question,
                            retrieved_facts,
                            answer_client,
                            answer_model,
                            answer_temperature=answer_temperature,
                            answer_max_tokens=answer_max_tokens,
                        )
                        qa_result = {
                            "question": question,
                            "answer": expected_answer,
                            "category": category,
                            "response": response,
                            "retrieved": retrieved_facts,
                            "retrieval_latency_seconds": retrieval_latency_seconds,
                        }
                    except Exception as e:
                        print(f"Error during search/answer for question {qa_idx + 1}/{len(qa_list)} '{question}': {e}")
                        raise RuntimeError(
                            f"Failed search/answer for sample {namespace}, question {qa_idx + 1}"
                        ) from e

                    sample_result["qa"].append(qa_result)
                    await save_checkpoint()
                    print(
                        f"Saved query checkpoint {len(sample_result['qa'])}/{len(qa_list)} "
                        f"for sample {namespace}."
                    )
    
                sample_result["_zep_complete"] = True
                processed_sample_ids.add(namespace)
    
                try:
                    summary, summary_path = await save_checkpoint()
                    avg_latency = summary.get("average_retrieval_latency_seconds")
                    avg_text = f", avg retrieval latency: {avg_latency:.3f}s" if avg_latency is not None else ""
                    print(
                        f"Results saved for sample {namespace}. "
                        f"Progress: {len(processed_sample_ids)}/{len(locomo_samples)} samples completed"
                        f"{avg_text}. Summary: {summary_path}"
                    )
                except Exception as e:
                    print(f"Warning: Could not save results for sample {namespace}: {e}")

        sample_outcomes = await asyncio.gather(
            *(
                process_sample(sample_idx, locomo_data)
                for sample_idx, locomo_data in enumerate(locomo_samples)
            ),
            return_exceptions=True,
        )
        sample_failures = [
            outcome for outcome in sample_outcomes if isinstance(outcome, BaseException)
        ]
        if sample_failures:
            raise RuntimeError(
                f"{len(sample_failures)} sample task(s) failed; checkpoints were preserved"
            ) from sample_failures[0]

        print(f"\n=== Finished processing all samples. Total samples in results: {len(all_results)} ===")
        summary, summary_path = save_results_and_summary(output_path, all_results)
        print(f"\nFinal results saved to: {output_path}")
        print(f"Run summary saved to: {summary_path}")
        total_questions = sum(len(sample["qa"]) for sample in all_results)
        print(f"Total questions processed: {total_questions}")
        avg_latency = summary.get("average_retrieval_latency_seconds")
        if avg_latency is not None:
            print(f"Average retrieval latency: {avg_latency:.3f}s over {summary['retrieval_latency_count']} queries")
        return all_results

    finally:
        if graphiti is not None:
            await graphiti.close()
            print("\nConnection closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Zep benchmark pipeline.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--token-file")
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(
        run_zep(
            dataset_path=args.dataset_path,
            output_path=args.output_path,
            embedding_base_url=args.embedding_base_url,
            token_file=args.token_file,
            retrieval_top_k=args.retrieval_top_k,
        )
    )
