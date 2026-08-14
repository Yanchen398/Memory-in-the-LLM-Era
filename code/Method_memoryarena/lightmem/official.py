import json
import fcntl
import multiprocessing
import os
import re
import shutil
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from .prompts import (
    ANSWER_PROMPT,
    ANSWER_PROMPT_StructMem,
    LoCoMo_Event_Binding_factual,
    LoCoMo_Event_Binding_relational,
    METADATA_GENERATE_PROMPT_locomo,
)
from ..dataset_hygiene import format_turn_text as shared_format_turn_text
from ..dataset_hygiene import natural_session_keys, raw_result_path


try:
    import yaml
except ImportError:  # pragma: no cover - YAML is optional for JSON-only configs.
    yaml = None


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

DEFAULT_OUTPUT_ROOT = os.path.abspath(os.path.join(CODE_DIR, "Result/LOCOMO"))
DEFAULT_LONGMEMEVAL_OUTPUT_ROOT = os.path.abspath(os.path.join(CODE_DIR, "Result/LONGMEMEVAL"))
DEFAULT_LLM_MODEL = "Qwen3.5-9B"
DEFAULT_LLM_API_KEY = "empty"
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLMLINGUA_MODEL_NAME = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
DEFAULT_LOCOMO_RETRIEVE_K = 10
DEFAULT_LONGMEMEVAL_RETRIEVE_K = 10
DEFAULT_RETRIEVE_TOP_KS = [10]
NO_THINKING_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

INIT_RESULT = {
    "add_input_prompt": [],
    "add_output_prompt": [],
    "api_call_nums": 0,
}


def no_thinking_extra_body() -> Dict:
    return {"chat_template_kwargs": {"enable_thinking": False}}


def uses_qwen_chat_template(model: str) -> bool:
    return "qwen" in str(model or "").lower()

def uses_responses_api(model: str) -> bool:
    return str(model or "").lower() == "gpt-5.4-mini"

def is_retryable_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
        return True
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }


def acquire_responses_api_slot():
    lock_dir = os.getenv("LIGHTMEM_RESPONSES_LOCK_DIR")
    if not lock_dir:
        return None
    max_concurrency = max(1, int(os.getenv("LIGHTMEM_RESPONSES_MAX_CONCURRENCY", "1")))
    os.makedirs(lock_dir, exist_ok=True)
    while True:
        for slot_index in range(max_concurrency):
            lock_file = open(os.path.join(lock_dir, f"slot_{slot_index}.lock"), "a+")
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_file
            except BlockingIOError:
                lock_file.close()
        time.sleep(0.05)


def release_responses_api_slot(lock_file):
    if lock_file is None:
        return
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()



def responses_create_with_retry(client, kwargs: Dict, max_attempts: int = 20):
    lock_file = acquire_responses_api_slot()
    try:
        for attempt in range(1, max_attempts + 1):
            try:
                return client.responses.create(**kwargs)
            except Exception as exc:
                if attempt >= max_attempts or not is_retryable_api_error(exc):
                    raise
                delay = min(2 ** (attempt - 1), 30) + ((os.getpid() % 10) / 10.0)
                print(
                    "Retrying Responses API call after transient error: "
                    f"attempt={attempt}/{max_attempts}, delay={delay:.1f}s, "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError("Responses API retry loop exited unexpectedly")
    finally:
        release_responses_api_slot(lock_file)




def parse_retrieve_top_ks(value, default_k: int) -> List[int]:
    if value is None or value == "":
        return [int(default_k)]
    if isinstance(value, int):
        raw_values = [value]
    elif isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",") if item.strip()]
    else:
        raw_values = list(value)

    top_ks = []
    for item in raw_values:
        k = int(item)
        if k <= 0:
            raise ValueError(f"retrieve_top_ks values must be positive, got {k}")
        if k not in top_ks:
            top_ks.append(k)
    return top_ks or [int(default_k)]


def empty_usage() -> Dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def merge_usage(*usages: Dict[str, int]) -> Dict[str, int]:
    merged = empty_usage()
    for usage in usages:
        for key in merged:
            merged[key] += int((usage or {}).get(key, 0) or 0)
    return merged


def latency_summary(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"count": 0, "average_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(latencies),
        "average_ms": float(np.mean(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }


def top_k_response_path(output_dir: str, top_k: int) -> str:
    return raw_result_path(os.path.join(output_dir, f"top_k_{top_k}"))


def top_k_token_path(output_dir: str, top_k: int) -> str:
    return os.path.join(output_dir, f"top_k_{top_k}", "token_tracker.json")


def write_status(status_path: str, payload: Dict):
    status_payload = dict(payload)
    status_payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(status_path, status_payload)

LOCOMO_ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def resolve_path(path_value: Optional[str], base_dir: str) -> Optional[str]:
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    candidates = [
        os.path.abspath(os.path.join(base_dir, path_value)),
        os.path.abspath(os.path.join(os.getcwd(), path_value)),
        os.path.abspath(os.path.join(CODE_DIR, path_value)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def safe_name(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))
    return sanitized.strip("._") or "sample"


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(path: str, payload):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def load_config_extras(config_path: Optional[str]) -> Dict:
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as file_obj:
        if config_path.endswith((".yaml", ".yml")):
            if yaml is None:
                return {}
            return yaml.safe_load(file_obj) or {}
        if config_path.endswith(".json"):
            return json.load(file_obj)
    return {}


def normalize_output_path(output_path: Optional[str], dataset_type: str) -> str:
    if output_path:
        return output_path if output_path.endswith(".json") else os.path.join(output_path, "result.json")
    root = DEFAULT_LONGMEMEVAL_OUTPUT_ROOT if dataset_type == "longmemeval" else DEFAULT_OUTPUT_ROOT
    return os.path.join(root, "lightmem", "official", "result.json")


def load_dataset(dataset_path: str) -> List[Dict]:
    data = read_json(dataset_path, [])
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "samples" in data and isinstance(data["samples"], list):
            return data["samples"]
        return [data]
    raise ValueError(f"Unsupported dataset format in {dataset_path}")


def detect_dataset_type(samples: List[Dict], configured_type: Optional[str] = None) -> str:
    if configured_type:
        normalized = configured_type.lower().replace("-", "").replace("_", "")
        if normalized in {"locomo"}:
            return "locomo"
        if normalized in {"longmemeval", "longmem"}:
            return "longmemeval"
    first = samples[0] if samples else {}
    if "haystack_sessions" in first and "question_id" in first:
        return "longmemeval"
    if "conversation" in first and "qa" in first:
        return "locomo"
    raise ValueError("Cannot infer dataset type. Set dataset_type to 'locomo' or 'longmemeval'.")


def apply_sample_slice(samples: List[Dict], start_idx: int, end_idx: Optional[int], ratio: float) -> List[Tuple[int, Dict]]:
    indexed_samples = list(enumerate(samples))
    sliced = indexed_samples[start_idx:end_idx] if end_idx is not None else indexed_samples[start_idx:]
    if ratio is None or ratio >= 1.0:
        return sliced
    if ratio <= 0:
        return []
    keep_count = max(1, int(len(sliced) * ratio)) if sliced else 0
    return sliced[:keep_count]


def parse_locomo_timestamp(timestamp_str: str) -> str:
    timestamp_str = str(timestamp_str or "").strip("() ")
    try:
        dt = datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return timestamp_str


def format_turn_text(turn: Dict) -> str:
    return shared_format_turn_text(turn)


def extract_locomo_sessions(conversation: Dict) -> Tuple[List[List[Dict]], List[str], str, str]:
    speaker_a = conversation.get("speaker_a", "Speaker_A")
    speaker_b = conversation.get("speaker_b", "Speaker_B")
    sessions = []
    timestamps = []
    for session_key in natural_session_keys(conversation):
        session_data = conversation.get(session_key, []) or []
        messages = []
        for turn in session_data:
            speaker_name = turn.get("speaker", "")
            speaker_id = "speaker_a" if speaker_name == speaker_a else "speaker_b"
            messages.append(
                {
                    "role": "user",
                    "content": format_turn_text(turn),
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                }
            )
        sessions.append(messages)
        timestamps.append(parse_locomo_timestamp(conversation.get(f"{session_key}_date_time", "")))
    return sessions, timestamps, speaker_a, speaker_b


def convert_locomo_longmemeval_sample(sample: Dict, source_record: Optional[Dict] = None) -> Dict:
    """Restore the native LongMemEval record shape without changing the input dataset file."""
    conversation = sample.get("conversation", {}) or {}
    speaker_a = str(conversation.get("speaker_a", "User")).strip().lower()
    speaker_b = str(conversation.get("speaker_b", "Assistant")).strip().lower()
    source_record = source_record or {}
    source_sessions = source_record.get("haystack_sessions", []) or []
    sessions = []
    timestamps = []
    for session_key in natural_session_keys(conversation):
        session_num = int(session_key.split("_")[1])
        restored_session = []
        source_session = source_sessions[session_num - 1] if session_num <= len(source_sessions) else []
        for turn_index, turn in enumerate(conversation.get(session_key, []) or []):
            speaker = str(turn.get("speaker", "")).strip().lower()
            if speaker == speaker_a or speaker == "user":
                role = "user"
            elif speaker == speaker_b or speaker == "assistant":
                role = "assistant"
            else:
                raise ValueError(
                    f"Cannot map converted LongMemEval speaker {turn.get('speaker')!r} "
                    f"in sample {sample.get('sample_id')}"
                )
            restored_turn = {
                "role": role,
                "content": str(turn.get("text", "")),
            }
            if turn_index < len(source_session):
                source_turn = source_session[turn_index]
                if (
                    source_turn.get("role") == role
                    and str(source_turn.get("content", "")) == restored_turn["content"]
                ):
                    restored_turn.update(
                        {key: value for key, value in source_turn.items() if key not in restored_turn}
                    )
            restored_session.append(restored_turn)
        sessions.append(restored_session)
        timestamps.append(str(conversation.get(f"{session_key}_date_time", "")))

    qa_items = sample.get("qa", []) or []
    if len(qa_items) != 1:
        raise ValueError(
            f"Converted LongMemEval sample {sample.get('sample_id')} must contain exactly one QA item"
        )
    qa = qa_items[0]
    source_record = source_record or {}
    return {
        "question_id": str(sample.get("sample_id", source_record.get("question_id", ""))),
        "question_type": qa.get("category", source_record.get("question_type", "")),
        "question": str(qa.get("question", source_record.get("question", ""))),
        "answer": str(qa.get("answer", source_record.get("answer", ""))),
        "question_date": source_record.get("question_date", sample.get("question_date", "")),
        "haystack_sessions": sessions,
        "haystack_dates": timestamps,
    }



def build_lightmem_config(
    collection_name: str,
    storage_dir: str,
    log_dir: str,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
    llm_provider: str,
    embedding_model_name: str,
    embedding_dim: int,
    embedding_device: str,
    embedding_base_url: Optional[str],
    embedding_api_key: str,
    pre_compress: bool,
    llmlingua_model_path: Optional[str],
    compression_rate: float,
    topic_segment: bool,
    metadata_generate: bool,
    text_summary: bool,
    extraction_mode: str,
    memory_manager_max_tokens: int,
) -> Dict:
    llmlingua_model_name = llmlingua_model_path or DEFAULT_LLMLINGUA_MODEL_NAME
    memory_manager_configs = {
        "model": llm_model,
        "api_key": llm_api_key,
        "temperature": 0.0,
        "max_tokens": memory_manager_max_tokens,
    }
    if uses_qwen_chat_template(llm_model):
        memory_manager_configs["extra_body"] = no_thinking_extra_body()
    if llm_provider == "openai":
        memory_manager_configs["openai_base_url"] = llm_base_url
    elif llm_provider == "deepseek":
        memory_manager_configs["deepseek_base_url"] = llm_base_url
    elif llm_provider == "vllm":
        memory_manager_configs["vllm_base_url"] = llm_base_url
    else:
        memory_manager_configs[f"{llm_provider}_base_url"] = llm_base_url

    config = {
        "pre_compress": pre_compress,
        "topic_segment": topic_segment,
        "precomp_topic_shared": pre_compress,
        "messages_use": "user_only",
        "metadata_generate": metadata_generate,
        "text_summary": text_summary,
        "memory_manager": {
            "model_name": llm_provider,
            "configs": memory_manager_configs,
        },
        "extract_threshold": 0.1,
        "index_strategy": "embedding",
        "text_embedder": {
            "model_name": "huggingface",
            "configs": {
                "model": embedding_model_name,
                "embedding_dims": embedding_dim,
                "model_kwargs": {"device": embedding_device},
                "huggingface_base_url": embedding_base_url,
                "api_key": embedding_api_key,
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": collection_name,
                "embedding_model_dims": embedding_dim,
                "path": os.path.join(storage_dir, collection_name),
                "on_disk": True,
            },
        },
        "summary_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": f"{collection_name}_summary",
                "embedding_model_dims": embedding_dim,
                "path": os.path.join(storage_dir, f"{collection_name}_summary"),
                "on_disk": True,
            },
        },
        "update": "offline",
        "logging": {
            "level": "INFO",
            "file_enabled": True,
            "log_dir": log_dir,
        },
        "extraction_mode": extraction_mode,
    }

    config["topic_segmenter"] = {
        "model_name": "llmlingua-2",
        "configs": {
            "model_name": llmlingua_model_name,
            "device_map": embedding_device,
            "use_llmlingua2": True,
        },
    }
    if pre_compress:
        config["pre_compressor"] = {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": llmlingua_model_name,
                    "device_map": embedding_device,
                    "use_llmlingua2": True,
                },
                "compress_config": {
                    "instruction": "",
                    "rate": compression_rate,
                    "target_token": -1,
                },
            },
        }
    else:
        config["pre_compressor"] = {"model_name": "entropy_compress"}
    return config


def summarize_token_stats(stats: Dict) -> Dict[str, int]:
    llm = stats.get("llm", {}) if isinstance(stats, dict) else {}
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    for stage in ("add_memory", "update", "summarize"):
        stage_stats = llm.get(stage, {}) or {}
        total_prompt += int(stage_stats.get("prompt_tokens", 0) or 0)
        total_completion += int(stage_stats.get("completion_tokens", 0) or 0)
        total_tokens += int(stage_stats.get("total_tokens", 0) or 0)
    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
    }


def extract_response_text(response) -> str:
    if not response:
        return ""
    output_text = getattr(response, "output_text", None)
    if output_text is not None:
        return str(output_text).strip()
    if not getattr(response, "choices", None):
        return ""
    content = response.choices[0].message.content
    return (content or "").strip()


def extract_usage(response) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    return {
        "prompt_tokens": prompt_tokens if prompt_tokens is not None else (getattr(usage, "input_tokens", 0) or 0),
        "completion_tokens": completion_tokens if completion_tokens is not None else (getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def add_usage(total: Dict[str, int], usage: Dict[str, int]):
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        total[key] = total.get(key, 0) + int(usage.get(key, 0) or 0)


def load_qdrant_entries(collection_name: str, qdrant_path: str, embedding_dim: int, with_vectors: bool = True) -> List[Dict]:
    from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig
    from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant

    cfg = QdrantConfig(
        collection_name=collection_name,
        path=qdrant_path,
        embedding_model_dims=embedding_dim,
        on_disk=True,
    )
    return Qdrant(cfg).get_all(with_vectors=with_vectors, with_payload=True)


def normalize_vector(vector):
    if isinstance(vector, dict):
        if "" in vector:
            return vector[""]
        if len(vector) == 1:
            return next(iter(vector.values()))
    return vector


class VectorRetriever:
    def __init__(self, embedder):
        self.embedder = embedder

    def retrieve(self, entries: List[Dict], query_text: str, limit: int) -> List[Dict]:
        query_vector = self.embedder.embed(query_text)
        results = []
        for entry in entries:
            vec = normalize_vector(entry.get("vector"))
            if vec is None:
                continue
            score = self._cosine_similarity(query_vector, vec)
            results.append(
                {
                    "id": str(entry.get("id")),
                    "score": float(score),
                    "payload": entry.get("payload", {}) or {},
                    "source": "vector",
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        a = np.array(v1)
        b = np.array(v2)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


def create_embedder(
    embedding_model_name: str,
    embedding_dim: int,
    embedding_device: str,
    embedding_base_url: Optional[str] = None,
    embedding_api_key: str = "EMPTY",
):
    from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig
    from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface

    embedder_cfg = BaseTextEmbedderConfig(
        model=embedding_model_name,
        embedding_dims=embedding_dim,
        model_kwargs={"device": embedding_device},
        huggingface_base_url=embedding_base_url,
        api_key=embedding_api_key,
    )
    return TextEmbedderHuggingface(embedder_cfg)


def format_related_memories(related: List[Dict]) -> str:
    out = []
    for item in related:
        payload = item.get("payload", {}) if isinstance(item, dict) else {}
        time_stamp = payload.get("time_stamp") or item.get("time_stamp") or ""
        weekday = payload.get("weekday") or item.get("weekday") or ""
        memory = (
            payload.get("memory")
            or payload.get("original_memory")
            or payload.get("compressed_memory")
            or item.get("memory")
            or ""
        )
        formatted_date = time_stamp
        try:
            formatted_date = datetime.fromisoformat(time_stamp.replace("Z", "+00:00")).strftime("%d %B %Y")
        except Exception:
            pass
        out.append(f"[Memory recorded on: {formatted_date}, {weekday}]\n{memory}".strip())
    return "\n\n".join(out)


def retrieve_locomo_entries(
    entries: List[Dict],
    retriever: VectorRetriever,
    question: str,
    retrieval_mode: str,
    total_limit: int,
    limit_per_speaker: int,
) -> List[Dict]:
    if retrieval_mode != "per-speaker":
        retrieved = retriever.retrieve(entries, question, total_limit)
        for entry in retrieved:
            entry["_retrieved_speaker"] = entry.get("payload", {}).get("speaker_name", "Unknown")
        return retrieved

    speaker_groups: Dict[str, List[Dict]] = {}
    for entry in entries:
        speaker = entry.get("payload", {}).get("speaker_name", "Unknown")
        speaker_groups.setdefault(speaker, []).append(entry)
    retrieved = []
    for speaker, group_entries in speaker_groups.items():
        speaker_retrieved = retriever.retrieve(group_entries, question, limit_per_speaker)
        for entry in speaker_retrieved:
            entry["_retrieved_speaker"] = speaker
        retrieved.extend(speaker_retrieved)
    return retrieved


def build_locomo_prompt(question: str, retrieved_entries: List[Dict], summaries: Optional[List[Dict]] = None) -> str:
    speaker_groups: Dict[str, List[Dict]] = {}
    for entry in retrieved_entries:
        speaker = entry.get("_retrieved_speaker") or entry.get("payload", {}).get("speaker_name", "Unknown")
        speaker_groups.setdefault(speaker, []).append(entry)
    speaker_names = list(speaker_groups.keys())
    if len(speaker_names) == 0:
        speaker_1_name, speaker_2_name = "Speaker 1", "Speaker 2"
        speaker_1_memories, speaker_2_memories = "No memories available.", "No memories available."
    elif len(speaker_names) == 1:
        speaker_1_name, speaker_2_name = speaker_names[0], "Speaker 2"
        speaker_1_memories = format_related_memories(speaker_groups[speaker_1_name])
        speaker_2_memories = "No memories available."
    else:
        speaker_1_name, speaker_2_name = speaker_names[0], speaker_names[1]
        speaker_1_memories = format_related_memories(speaker_groups[speaker_1_name])
        speaker_2_memories = format_related_memories(speaker_groups[speaker_2_name])

    if summaries is not None:
        session_summaries = "\n".join(
            (item.get("payload", {}) or {}).get("summary")
            or (item.get("payload", {}) or {}).get("memory", "")
            for item in summaries
        ) or "No session summaries available."
        return ANSWER_PROMPT_StructMem.format(
            speaker_1_name=speaker_1_name,
            speaker_1_memories=speaker_1_memories,
            speaker_2_name=speaker_2_name,
            speaker_2_memories=speaker_2_memories,
            session_summaries=session_summaries,
            question=question,
        )

    return ANSWER_PROMPT.format(
        speaker_1_name=speaker_1_name,
        speaker_1_memories=speaker_1_memories,
        speaker_2_name=speaker_2_name,
        speaker_2_memories=speaker_2_memories,
        question=question,
    )


def extract_json_label(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text).get("label", "")
    except Exception:
        return text


def evaluate_locomo_judge(client, model: str, question: str, gold_answer: str, generated_answer: str) -> Tuple[int, Dict[str, int]]:
    response = chat_completion(
        client,
        model,
        [
            {
                "role": "user",
                "content": LOCOMO_ACCURACY_PROMPT.format(
                    question=question,
                    gold_answer=gold_answer,
                    generated_answer=generated_answer,
                ),
            }
        ],
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    label = extract_json_label(extract_response_text(response))
    return (1 if label == "CORRECT" else 0), extract_usage(response)


def get_longmemeval_anscheck_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == "temporal-reasoning":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == "knowledge-update":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        elif task == "single-session-preference":
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
        else:
            raise NotImplementedError(f"Unsupported LongMemEval question type: {task}")
        return template.format(question, answer, response)
    template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    return template.format(question, answer, response)


def true_or_false(response: str) -> bool:
    normalized = str(response or "").strip().lower()
    if not normalized:
        return False
    first_line = normalized.splitlines()[0].strip()
    tokens = first_line.replace(".", "").replace("!", "").replace(":", "").replace(";", "").split()
    if not tokens:
        return False
    if tokens[0] in {"yes", "y"}:
        return True
    if tokens[0] in {"no", "n"}:
        return False
    if "yes" in first_line:
        return True
    if "no" in first_line:
        return False
    return False


def chat_completion(
    client,
    model: str,
    messages: List[Dict],
    max_tokens: Optional[int] = 2000,
    top_p: Optional[float] = None,
    response_format: Optional[Dict] = None,
):
    if uses_responses_api(model):
        kwargs = {"model": model, "input": messages}
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        if response_format:
            kwargs["text"] = {"format": response_format}
        return responses_create_with_retry(client, kwargs)

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "stream": False,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if top_p is not None:
        kwargs["top_p"] = top_p
    if response_format:
        kwargs["response_format"] = response_format
    if uses_qwen_chat_template(model):
        kwargs["extra_body"] = no_thinking_extra_body()
    return client.chat.completions.create(**kwargs)


def aggregate_token_stats(token_file: str, method: str, sample_stats: List[Dict]):
    payload = {
        "method": method,
        "sample_count": len(sample_stats),
        "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in sample_stats),
        "completion_tokens": sum(item.get("completion_tokens", 0) for item in sample_stats),
        "total_tokens": sum(item.get("total_tokens", 0) for item in sample_stats),
        "samples": sample_stats,
    }
    write_json(token_file, payload)


def make_clients(extras: Dict, llm_api_key: str, llm_base_url: str, llm_model: str):
    from openai import OpenAI

    answer_client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
    judge_model = extras.get("judge_model") or extras.get("llm_judge_model") or llm_model
    judge_api_key = extras.get("judge_api_key") or extras.get("llm_judge_api_key") or llm_api_key
    judge_base_url = extras.get("judge_base_url") or extras.get("llm_judge_base_url") or llm_base_url
    judge_client = OpenAI(api_key=judge_api_key, base_url=judge_base_url)
    return answer_client, judge_client, judge_model


def build_lightmem_instance(
    collection_name: str,
    storage_dir: str,
    log_dir: str,
    params: Dict,
):
    from lightmem.memory.lightmem import LightMemory

    config = build_lightmem_config(
        collection_name=collection_name,
        storage_dir=storage_dir,
        log_dir=log_dir,
        llm_model=params["llm_model"],
        llm_api_key=params["llm_api_key"],
        llm_base_url=params["llm_base_url"],
        embedding_base_url=params.get("embedding_base_url"),
        embedding_api_key=params.get("embedding_api_key", "EMPTY"),
        llm_provider=params["llm_provider"],
        embedding_model_name=params["embedding_model_name"],
        embedding_dim=params["embedding_dim"],
        embedding_device=params["embedding_device"],
        pre_compress=params["pre_compress"],
        llmlingua_model_path=params.get("llmlingua_model_path"),
        compression_rate=params["compression_rate"],
        topic_segment=params["topic_segment"],
        metadata_generate=params["metadata_generate"],
        text_summary=params["text_summary"],
        extraction_mode=params["extraction_mode"],
        memory_manager_max_tokens=params["memory_manager_max_tokens"],
    )
    return LightMemory.from_config(config)


def process_locomo_sample(
    sample: Dict,
    dataset_index: int,
    output_dir: str,
    memory_root: str,
    log_root: str,
    params: Dict,
    extras: Dict,
) -> Tuple[Dict[str, Dict], Dict]:
    collection_name = safe_name(str(sample.get("sample_id", dataset_index)))
    sample_runtime = os.path.join(memory_root, f"{dataset_index:05d}_{collection_name}")
    post_root = os.path.join(sample_runtime, "qdrant_post_update")
    pre_root = os.path.join(sample_runtime, "qdrant_pre_update")
    log_dir = os.path.join(log_root, f"{dataset_index:05d}_{collection_name}")

    if os.path.exists(sample_runtime):
        shutil.rmtree(sample_runtime)
    os.makedirs(post_root, exist_ok=True)
    os.makedirs(pre_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    lightmem = build_lightmem_instance(collection_name, post_root, log_dir, params)
    prompt_arg = (
        {
            "factual": LoCoMo_Event_Binding_factual,
            "relational": LoCoMo_Event_Binding_relational,
        }
        if params["extraction_mode"] == "event"
        else METADATA_GENERATE_PROMPT_locomo
    )

    sessions, timestamps, speaker_a, speaker_b = extract_locomo_sessions(sample.get("conversation", {}) or {})
    for session, timestamp in zip(sessions, timestamps):
        while session and session[0]["role"] != "user":
            session.pop(0)
        num_turns = len(session) // 2
        for turn_idx in range(num_turns):
            turn_messages = session[turn_idx * 2 : turn_idx * 2 + 2]
            if len(turn_messages) < 2 or turn_messages[0]["role"] != "user" or turn_messages[1]["role"] != "assistant":
                continue
            for msg in turn_messages:
                msg["time_stamp"] = timestamp
            is_last_turn = session is sessions[-1] and turn_idx == num_turns - 1
            lightmem.add_memory(
                messages=turn_messages,
                METADATA_GENERATE_PROMPT=prompt_arg,
                force_segment=is_last_turn,
                force_extract=is_last_turn,
            )

    source_dir = os.path.join(post_root, collection_name)
    backup_dir = os.path.join(pre_root, collection_name)
    if os.path.exists(source_dir):
        shutil.copytree(source_dir, backup_dir)

    summary_count = 0
    if params["enable_summary"]:
        summary_mem = build_lightmem_instance(collection_name, pre_root, log_dir, params)
        summary_result = summary_mem.summarize(
            retrieval_scope="global",
            time_window=params["summary_time_window"],
            top_k_seeds=params["summary_top_k_seeds"],
            process_all=True,
        )
        summary_count = int(summary_result.get("total_summaries", 0) or 0) if isinstance(summary_result, dict) else 0

    if params["enable_update"]:
        lightmem.construct_update_queue_all_entries(
            top_k=params["update_top_k"],
            keep_top_n=params["update_keep_top_n"],
            max_workers=params["update_workers"],
        )
        lightmem.offline_update_all_entries(
            score_threshold=params["update_score_threshold"],
            max_workers=params["update_workers"],
        )

    retrieval_state = str(extras.get("retrieval_memory_state", "pre")).lower()
    retrieval_root = post_root if retrieval_state == "post" else pre_root
    retrieval_collection_path = os.path.join(retrieval_root, collection_name)

    entries = load_qdrant_entries(collection_name, retrieval_collection_path, params["embedding_dim"], with_vectors=True)
    summaries = []
    if params["enable_summary"]:
        summary_path = os.path.join(pre_root, f"{collection_name}_summary")
        if os.path.exists(summary_path):
            summaries = load_qdrant_entries(f"{collection_name}_summary", summary_path, params["embedding_dim"], with_vectors=True)

    embedder = create_embedder(
        params["embedding_model_name"],
        params["embedding_dim"],
        params["embedding_device"],
        params.get("embedding_base_url"),
        params.get("embedding_api_key", "EMPTY"),
    )
    retriever = VectorRetriever(embedder)
    answer_client, judge_client, judge_model = make_clients(
        extras,
        params["llm_api_key"],
        params["llm_base_url"],
        params["llm_model"],
    )

    retrieval_mode = extras.get("retrieval_mode", "combined")
    retrieve_top_ks = params.get("retrieve_top_ks") or [params["retrieve_k"]]
    allow_categories = {int(item) for item in extras.get("allow_categories", [1, 2, 3, 4])}
    enable_judge = bool(extras.get("enable_judge", True))
    summary_limit = int(extras.get("summary_limit", 5))

    qa_results_by_k = {int(top_k): [] for top_k in retrieve_top_ks}
    answer_tokens_by_k = {int(top_k): empty_usage() for top_k in retrieve_top_ks}
    judge_tokens_by_k = {int(top_k): empty_usage() for top_k in retrieve_top_ks}
    retrieval_latencies_by_k = {int(top_k): [] for top_k in retrieve_top_ks}

    for qa in sample.get("qa", []):
        category = qa.get("category")
        try:
            category_id = int(category)
        except Exception:
            category_id = category
        if isinstance(category_id, int) and (category_id == 5 or category_id not in allow_categories):
            continue

        question = qa.get("question", "")
        reference = qa.get("answer") or qa.get("adversarial_answer", "")

        for top_k in retrieve_top_ks:
            top_k = int(top_k)
            retrieval_started = time.perf_counter()
            retrieved_entries = retrieve_locomo_entries(
                entries,
                retriever,
                question,
                retrieval_mode=retrieval_mode,
                total_limit=top_k,
                limit_per_speaker=top_k,
            )
            retrieved_summaries = retriever.retrieve(summaries, question, summary_limit) if summaries else None
            retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000.0
            retrieval_latencies_by_k[top_k].append(retrieval_latency_ms)

            prompt = build_locomo_prompt(question, retrieved_entries, retrieved_summaries)
            response_obj = chat_completion(
                answer_client,
                params["llm_model"],
                [{"role": "user", "content": prompt}],
                max_tokens=None,
            )
            response = extract_response_text(response_obj)
            usage = extract_usage(response_obj)
            add_usage(answer_tokens_by_k[top_k], usage)

            result = {
                "question": question,
                "answer": reference,
                "reference": reference,
                "category": category,
                "response": response,
                "prediction": response,
                "retrieved": [format_related_memories([entry]) for entry in retrieved_entries],
                "retrieved_count": len(retrieved_entries),
                "retrieve_k": top_k,
                "retrieval_latency_ms": retrieval_latency_ms,
                "search_duration_ms": retrieval_latency_ms,
            }
            if enable_judge:
                if str(extras.get("eval_dataset", "loco")).lower() == "lme":
                    judge_prompt = get_longmemeval_anscheck_prompt(
                        str(category),
                        question,
                        reference,
                        response,
                        abstention="abs" in str(sample.get("sample_id", "")),
                    )
                    judge_response = chat_completion(
                        judge_client,
                        judge_model,
                        [{"role": "user", "content": judge_prompt}],
                        max_tokens=int(extras.get("judge_max_tokens", 2000)),
                        top_p=float(extras.get("judge_top_p", 0.8)),
                    )
                    judge_usage = extract_usage(judge_response)
                    correct = 1 if true_or_false(extract_response_text(judge_response)) else 0
                else:
                    correct, judge_usage = evaluate_locomo_judge(
                        judge_client,
                        judge_model,
                        question,
                        reference,
                        response,
                    )
                add_usage(judge_tokens_by_k[top_k], judge_usage)
                result["metrics"] = {"judge_correct": correct}
                result["correct"] = correct
            qa_results_by_k[top_k].append(result)

    sample_id = str(sample.get("sample_id", collection_name))
    sample_results_by_k = {}
    top_k_tokens = {}
    for top_k in retrieve_top_ks:
        top_k = int(top_k)
        sample_results_by_k[str(top_k)] = {
            "sample_id": sample_id,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "memory_state_for_retrieval": retrieval_state,
            "summary_count": summary_count,
            "retrieve_k": top_k,
            "average_retrieval_latency_ms": latency_summary(retrieval_latencies_by_k[top_k])["average_ms"],
            "qa": qa_results_by_k[top_k],
        }
        top_k_usage = merge_usage(answer_tokens_by_k[top_k], judge_tokens_by_k[top_k])
        top_k_tokens[str(top_k)] = {
            **top_k_usage,
            "answer_tokens": answer_tokens_by_k[top_k],
            "judge_tokens": judge_tokens_by_k[top_k],
            "retrieval_latency_ms": latency_summary(retrieval_latencies_by_k[top_k]),
        }

    memory_tokens = summarize_token_stats(lightmem.get_token_statistics())
    answer_tokens_total = empty_usage()
    judge_tokens_total = empty_usage()
    for top_k in retrieve_top_ks:
        add_usage(answer_tokens_total, answer_tokens_by_k[int(top_k)])
        add_usage(judge_tokens_total, judge_tokens_by_k[int(top_k)])

    sample_tokens = {
        **merge_usage(memory_tokens, answer_tokens_total, judge_tokens_total),
        "memory_tokens": memory_tokens,
        "answer_tokens": answer_tokens_total,
        "judge_tokens": judge_tokens_total,
        "top_k_tokens": top_k_tokens,
    }
    return sample_results_by_k, sample_tokens

def process_longmemeval_sample(
    sample: Dict,
    dataset_index: int,
    memory_root: str,
    log_root: str,
    params: Dict,
    extras: Dict,
) -> Tuple[Dict, Dict]:
    collection_name = safe_name(str(sample.get("question_id", dataset_index)))
    sample_runtime = os.path.join(memory_root, f"{dataset_index:05d}_{collection_name}")
    log_dir = os.path.join(log_root, f"{dataset_index:05d}_{collection_name}")
    if os.path.exists(sample_runtime):
        shutil.rmtree(sample_runtime)
    os.makedirs(sample_runtime, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    lightmem = build_lightmem_instance(collection_name, sample_runtime, log_dir, params)
    results_list = []
    construction_start = time.time()
    sessions = sample.get("haystack_sessions", []) or []
    timestamps = sample.get("haystack_dates", []) or []
    for session_index, (session, timestamp) in enumerate(zip(sessions, timestamps)):
        session = list(session)
        while session and session[0].get("role") != "user":
            session.pop(0)
        num_turns = len(session) // 2
        for turn_idx in range(num_turns):
            turn_messages = session[turn_idx * 2 : turn_idx * 2 + 2]
            if len(turn_messages) < 2 or turn_messages[0].get("role") != "user" or turn_messages[1].get("role") != "assistant":
                continue
            turn_messages = [dict(msg) for msg in turn_messages]
            for msg in turn_messages:
                msg["time_stamp"] = timestamp
            is_last_turn = session_index == len(sessions) - 1 and turn_idx == num_turns - 1
            result = lightmem.add_memory(
                messages=turn_messages,
                force_segment=is_last_turn,
                force_extract=is_last_turn,
            )
            if result != INIT_RESULT:
                results_list.append(result)
    construction_time = time.time() - construction_start

    answer_client, judge_client, judge_model = make_clients(
        extras,
        params["llm_api_key"],
        params["llm_base_url"],
        params["llm_model"],
    )
    retrieve_k = int(params["retrieve_k"] or DEFAULT_LONGMEMEVAL_RETRIEVE_K)
    retrieval_started = time.perf_counter()
    related_memories = lightmem.retrieve(sample.get("question", ""), limit=retrieve_k)
    retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000.0
    related_memories_text = "\n".join(related_memories)
    user_prompt = (
        f"Question time:{sample.get('question_date')} and question:{sample.get('question')}\n"
        f"Please answer the question based on the following memories: {related_memories_text}"
    )
    response_obj = chat_completion(
        answer_client,
        params["llm_model"],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=int(extras.get("answer_max_tokens", 2000)),
        top_p=float(extras.get("answer_top_p", 0.8)),
    )
    generated_answer = extract_response_text(response_obj)
    answer_tokens = extract_usage(response_obj)
    judge_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    enable_judge = bool(extras.get("enable_judge", True))
    correct = None
    if enable_judge:
        judge_prompt = get_longmemeval_anscheck_prompt(
            sample.get("question_type", ""),
            sample.get("question", ""),
            sample.get("answer", ""),
            generated_answer,
            abstention="abs" in str(sample.get("question_id", "")),
        )
        judge_response = chat_completion(
            judge_client,
            judge_model,
            [{"role": "user", "content": judge_prompt}],
            max_tokens=int(extras.get("judge_max_tokens", 2000)),
            top_p=float(extras.get("judge_top_p", 0.8)),
        )
        judge_text = extract_response_text(judge_response)
        judge_tokens = extract_usage(judge_response)
        correct = 1 if true_or_false(judge_text) else 0

    memory_tokens = summarize_token_stats(lightmem.get_token_statistics())
    sample_tokens = {
        "prompt_tokens": memory_tokens["prompt_tokens"] + answer_tokens["prompt_tokens"] + judge_tokens["prompt_tokens"],
        "completion_tokens": memory_tokens["completion_tokens"] + answer_tokens["completion_tokens"] + judge_tokens["completion_tokens"],
        "total_tokens": memory_tokens["total_tokens"] + answer_tokens["total_tokens"] + judge_tokens["total_tokens"],
        "memory_tokens": memory_tokens,
        "answer_tokens": answer_tokens,
        "judge_tokens": judge_tokens,
    }
    qa_item = {
        "question": sample.get("question", ""),
        "answer": sample.get("answer", ""),
        "category": sample.get("question_type", ""),
        "response": generated_answer,
        "retrieved": related_memories,
        "construction_time": construction_time,
        "retrieve_k": retrieve_k,
        "retrieval_latency_ms": retrieval_latency_ms,
        "search_duration_ms": retrieval_latency_ms,
    }
    if correct is not None:
        qa_item["correct"] = correct
        qa_item["metrics"] = {"judge_correct": correct}
    return (
        {
            "sample_id": str(sample.get("question_id", collection_name)),
            "question_id": sample.get("question_id", collection_name),
            "results": results_list,
            "construction_time": construction_time,
            "generated_answer": generated_answer,
            "ground_truth": sample.get("answer", ""),
            "correct": correct,
            "qa": [qa_item],
        },
        sample_tokens,
    )


def process_converted_longmemeval_sample(
    sample: Dict,
    source_record: Optional[Dict],
    dataset_index: int,
    memory_root: str,
    log_root: str,
    params: Dict,
    extras: Dict,
) -> Tuple[Dict, Dict]:
    restored_sample = convert_locomo_longmemeval_sample(sample, source_record)
    return process_longmemeval_sample(
        sample=restored_sample,
        dataset_index=dataset_index,
        memory_root=memory_root,
        log_root=log_root,
        params=params,
        extras=extras,
    )



def build_params(
    extras: Dict,
    llm_model: Optional[str],
    llm_api_key: Optional[str],
    llm_base_url: Optional[str],
    llm_provider: Optional[str],
    embedding_model_name: Optional[str],
    embedding_dim: int,
    embedding_device: str,
    retrieve_k: Optional[int],
    retrieve_top_ks,
    pre_compress: bool,
    llmlingua_model_path: Optional[str],
    compression_rate: float,
    topic_segment: bool,
    metadata_generate: bool,
    text_summary: bool,
    extraction_mode: str,
    enable_update: bool,
    update_top_k: int,
    update_keep_top_n: int,
    update_score_threshold: float,
    update_workers: int,
    enable_summary: bool,
    summary_time_window: int,
    summary_top_k_seeds: int,
    dataset_type: str,
) -> Dict:
    official_defaults = bool(extras.get("official_defaults", False))
    default_k = DEFAULT_LONGMEMEVAL_RETRIEVE_K if dataset_type == "longmemeval" else DEFAULT_LOCOMO_RETRIEVE_K
    effective_retrieve_k = retrieve_k
    if official_defaults and "retrieve_k" not in extras:
        effective_retrieve_k = default_k
    effective_retrieve_k = int(effective_retrieve_k or default_k)
    effective_retrieve_top_ks = parse_retrieve_top_ks(
        retrieve_top_ks if retrieve_top_ks is not None else extras.get("retrieve_top_ks"),
        effective_retrieve_k,
    )
    effective_pre_compress = pre_compress
    if official_defaults and "pre_compress" not in extras:
        effective_pre_compress = True
    effective_embedding_device = embedding_device
    if official_defaults and "embedding_device" not in extras:
        effective_embedding_device = "cuda"
    return {
        "llm_model": llm_model or DEFAULT_LLM_MODEL,
        "llm_api_key": llm_api_key or DEFAULT_LLM_API_KEY,
        "llm_base_url": llm_base_url or DEFAULT_LLM_BASE_URL,
        "llm_provider": llm_provider or "openai",
        "embedding_model_name": embedding_model_name or DEFAULT_EMBEDDING_MODEL_NAME,
        "embedding_dim": embedding_dim or 384,
        "embedding_device": effective_embedding_device or ("cuda" if official_defaults else "cpu"),
        "embedding_base_url": extras.get("embedding_base_url"),
        "embedding_api_key": extras.get("embedding_api_key", "EMPTY"),
        "retrieve_k": effective_retrieve_k,
        "retrieve_top_ks": effective_retrieve_top_ks,
        "pre_compress": bool(effective_pre_compress if effective_pre_compress is not None else official_defaults),
        "llmlingua_model_path": resolve_path(llmlingua_model_path, os.getcwd()) if llmlingua_model_path else None,
        "compression_rate": compression_rate if compression_rate is not None else 0.6,
        "topic_segment": bool(topic_segment),
        "metadata_generate": bool(metadata_generate),
        "text_summary": bool(text_summary),
        "extraction_mode": extraction_mode or "flat",
        "memory_manager_max_tokens": int(
            extras.get("memory_manager_max_tokens", 16000 if dataset_type == "longmemeval" else 4096)
        ),
        "enable_update": bool(enable_update) if dataset_type == "locomo" else bool(extras.get("longmemeval_enable_update", False)),
        "update_top_k": update_top_k if update_top_k is not None else 20,
        "update_keep_top_n": update_keep_top_n if update_keep_top_n is not None else 10,
        "update_score_threshold": update_score_threshold if update_score_threshold is not None else (0.9 if dataset_type == "locomo" else 0.8),
        "update_workers": update_workers if update_workers is not None else 5,
        "enable_summary": bool(enable_summary),
        "summary_time_window": summary_time_window if summary_time_window is not None else 3600,
        "summary_top_k_seeds": summary_top_k_seeds if summary_top_k_seeds is not None else 15,
    }


def initialize_sample_worker(requested_cuda_devices: int):
    if requested_cuda_devices <= 0:
        return

    import torch

    available_cuda_devices = torch.cuda.device_count()
    cuda_device_count = min(requested_cuda_devices, available_cuda_devices)
    if cuda_device_count <= 0:
        return

    worker_identity = multiprocessing.current_process()._identity
    worker_number = worker_identity[0] if worker_identity else 1
    cuda_device = (worker_number - 1) % cuda_device_count
    torch.cuda.set_device(cuda_device)
    print(
        f"Initialized {multiprocessing.current_process().name} on logical CUDA device {cuda_device}",
        flush=True,
    )


def process_sample_worker(task: Dict) -> Dict:
    dataset_type = task["dataset_type"]
    execution_pipeline = task.get("execution_pipeline", dataset_type)
    dataset_index = task["dataset_index"]
    sample = task["sample"]
    sample_id = task["sample_id"]
    cache_file = task["cache_file"]
    params = task["params"]
    force_rebuild = task["force_rebuild"]

    if os.path.exists(cache_file) and not force_rebuild:
        cached = read_json(cache_file, {})
        if execution_pipeline == "locomo" and isinstance(cached, dict) and "results_by_top_k" in cached:
            return {
                "dataset_index": dataset_index,
                "sample_id": sample_id,
                "sample_results_by_k": cached.get("results_by_top_k", {}),
                "sample_tokens": cached.get("sample_tokens"),
                "cached": True,
            }
        if execution_pipeline in {"longmemeval", "longmemeval_converted"} and isinstance(cached, dict) and cached.get("qa"):
            return {
                "dataset_index": dataset_index,
                "sample_id": sample_id,
                "sample_results_by_k": {str(params["retrieve_k"]): cached},
                "sample_tokens": None,
                "cached": True,
            }

    try:
        if execution_pipeline == "longmemeval":
            sample_result, sample_tokens = process_longmemeval_sample(
                sample=sample,
                dataset_index=dataset_index,
                memory_root=task["memory_root"],
                log_root=task["log_dir"],
                params=params,
                extras=task["extras"],
            )
            sample_results_by_k = {str(params["retrieve_k"]): sample_result}
            write_json(cache_file, sample_result)
        elif execution_pipeline == "longmemeval_converted":
            sample_result, sample_tokens = process_converted_longmemeval_sample(
                sample=sample,
                source_record=task.get("longmemeval_source_record"),
                dataset_index=dataset_index,
                memory_root=task["memory_root"],
                log_root=task["log_dir"],
                params=params,
                extras=task["extras"],
            )
            if not task["track_tokens"]:
                sample_tokens = None
            sample_results_by_k = {str(params["retrieve_k"]): sample_result}
            write_json(cache_file, sample_result)
        else:
            sample_results_by_k, sample_tokens = process_locomo_sample(
                sample=sample,
                dataset_index=dataset_index,
                output_dir=task["output_dir"],
                memory_root=task["memory_root"],
                log_root=task["log_dir"],
                params=params,
                extras=task["extras"],
            )
            if not task["track_tokens"]:
                sample_tokens = None
            write_json(
                cache_file,
                {"sample_id": sample_id, "results_by_top_k": sample_results_by_k, "sample_tokens": sample_tokens},
            )
    except Exception as exc:
        raise RuntimeError(f"LightMem sample {sample_id} (index {dataset_index}) failed") from exc

    return {
        "dataset_index": dataset_index,
        "sample_id": sample_id,
        "sample_results_by_k": sample_results_by_k,
        "sample_tokens": sample_tokens,
        "cached": False,
    }


def run_lightmem_official(
    dataset_path,
    output_path=None,
    memory_path=None,
    token_file=None,
    llm_model=DEFAULT_LLM_MODEL,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    llm_provider="openai",
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_dim=384,
    embedding_device="cpu",
    retrieve_k=None,
    retrieve_top_ks=None,
    ratio=1.0,
    start_idx=0,
    end_idx=None,
    pre_compress=False,
    llmlingua_model_path=None,
    compression_rate=0.6,
    topic_segment=True,
    metadata_generate=True,
    text_summary=True,
    extraction_mode="flat",
    enable_update=True,
    update_top_k=20,
    update_keep_top_n=10,
    update_score_threshold=0.9,
    update_workers=5,
    enable_summary=False,
    summary_time_window=3600,
    summary_top_k_seeds=15,
    auto_eval=None,
    eval_embedding_model=None,
    eval_result_path=None,
    config_path=None,
):
    extras = load_config_extras(config_path)
    llmlingua_service_url = str(extras.get("llmlingua_service_url", "")).rstrip("/")
    if llmlingua_service_url:
        os.environ["LIGHTMEM_LLMLINGUA_SERVICE_URL"] = llmlingua_service_url
    if auto_eval is not None:
        extras["auto_eval"] = auto_eval
    if eval_embedding_model:
        extras["eval_embedding_model"] = eval_embedding_model
    if eval_result_path:
        extras["eval_result_path"] = eval_result_path
    base_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()
    dataset_path = resolve_path(dataset_path, base_dir)
    if not dataset_path:
        raise ValueError("LightMem requires 'dataset_path' in the config or CLI arguments.")

    all_samples = load_dataset(dataset_path)
    dataset_type = detect_dataset_type(all_samples, extras.get("dataset_type"))
    execution_pipeline = dataset_type
    if dataset_type == "locomo" and str(extras.get("eval_dataset", "loco")).lower() == "lme":
        execution_pipeline = "longmemeval_converted"

    longmemeval_source_by_id = {}
    if execution_pipeline == "longmemeval_converted":
        source_path = resolve_path(extras.get("longmemeval_source_path"), base_dir)
        if source_path:
            for source_record in load_dataset(source_path):
                question_id = str(source_record.get("question_id", ""))
                if question_id:
                    longmemeval_source_by_id[question_id] = source_record
        missing_question_dates = [
            str(sample.get("sample_id"))
            for sample in all_samples
            if str(sample.get("sample_id")) not in longmemeval_source_by_id
            and not sample.get("question_date")
        ]
        if missing_question_dates:
            raise ValueError(
                "Converted LongMemEval requires question_date metadata; missing samples: "
                + ", ".join(missing_question_dates[:5])
            )
    output_path = normalize_output_path(resolve_path(output_path, base_dir), dataset_type)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    memory_root = resolve_path(memory_path, base_dir) or os.path.join(output_dir, "_lightmem_official_runtime")
    cache_dir = os.path.join(output_dir, "_lightmem_official_cache")
    log_dir = os.path.join(output_dir, "_lightmem_official_logs")
    os.makedirs(memory_root, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    token_file = resolve_path(token_file, base_dir) or os.path.join(output_dir, "token_tracker.json")
    status_path = os.path.join(output_dir, "run_status.json")

    params = build_params(
        extras=extras,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_provider=llm_provider,
        embedding_model_name=embedding_model_name,
        embedding_dim=embedding_dim,
        embedding_device=embedding_device,
        retrieve_k=retrieve_k,
        retrieve_top_ks=retrieve_top_ks,
        pre_compress=pre_compress,
        llmlingua_model_path=llmlingua_model_path,
        compression_rate=compression_rate,
        topic_segment=topic_segment,
        metadata_generate=metadata_generate,
        text_summary=text_summary,
        extraction_mode=extraction_mode,
        enable_update=enable_update,
        update_top_k=update_top_k,
        update_keep_top_n=update_keep_top_n,
        update_score_threshold=update_score_threshold,
        update_workers=update_workers,
        enable_summary=enable_summary,
        summary_time_window=summary_time_window,
        summary_top_k_seeds=summary_top_k_seeds,
        dataset_type="longmemeval" if execution_pipeline == "longmemeval_converted" else dataset_type,
    )

    selected_samples = apply_sample_slice(
        all_samples,
        start_idx=start_idx if start_idx is not None else 0,
        end_idx=end_idx,
        ratio=ratio if ratio is not None else 1.0,
    )

    retrieve_top_ks = params["retrieve_top_ks"] if dataset_type == "locomo" else [params["retrieve_k"]]
    results_by_k = {str(top_k): [] for top_k in retrieve_top_ks}
    token_stats = []
    token_stats_by_k = {str(top_k): [] for top_k in retrieve_top_ks}
    force_rebuild = bool(extras.get("force_rebuild", False))
    track_tokens = bool(extras.get("track_tokens", True))
    configured_processes = max(1, int(extras.get("num_processes", 1)))
    num_processes = min(configured_processes, len(selected_samples)) if selected_samples else 1
    worker_cuda_device_count = max(0, int(extras.get("worker_cuda_device_count", 1)))
    if not track_tokens:
        token_file = None

    top_k_outputs = {str(top_k): top_k_response_path(output_dir, int(top_k)) for top_k in retrieve_top_ks}
    status_common = {
        "dataset_type": dataset_type,
        "execution_pipeline": execution_pipeline,
        "total_samples": len(selected_samples),
        "retrieve_top_ks": retrieve_top_ks,
        "num_processes": num_processes,
        "worker_cuda_device_count": worker_cuda_device_count,
        "llmlingua_service_url": llmlingua_service_url or None,
        "track_tokens": track_tokens,
        "output_dir": output_dir,
        "top_k_outputs": top_k_outputs,
    }
    write_status(status_path, {"status": "running", "processed_samples": 0, **status_common})

    tasks = []
    for dataset_index, sample in selected_samples:
        sample_id = str(sample.get("sample_id") or sample.get("question_id") or dataset_index)
        tasks.append(
            {
                "dataset_type": dataset_type,
                "execution_pipeline": execution_pipeline,
                "dataset_index": dataset_index,
                "sample": sample,
                "sample_id": sample_id,
                "longmemeval_source_record": longmemeval_source_by_id.get(sample_id),
                "cache_file": os.path.join(cache_dir, f"{dataset_index:05d}_{safe_name(sample_id)}.json"),
                "output_dir": output_dir,
                "memory_root": memory_root,
                "log_dir": log_dir,
                "params": params,
                "extras": extras,
                "force_rebuild": force_rebuild,
                "track_tokens": track_tokens,
            }
        )

    completed_by_index = {}

    def rebuild_aggregates():
        rebuilt_results = {str(top_k): [] for top_k in retrieve_top_ks}
        rebuilt_tokens = []
        rebuilt_tokens_by_k = {str(top_k): [] for top_k in retrieve_top_ks}
        for selected_index, _ in selected_samples:
            completed = completed_by_index.get(selected_index)
            if completed is None:
                continue
            sample_id = completed["sample_id"]
            sample_results_by_k = completed["sample_results_by_k"]
            sample_tokens = completed.get("sample_tokens")
            for top_k in retrieve_top_ks:
                key = str(top_k)
                if key in sample_results_by_k:
                    rebuilt_results[key].append(sample_results_by_k[key])
            if track_tokens and sample_tokens:
                rebuilt_tokens.append({"sample_id": sample_id, **sample_tokens})
                for top_k in retrieve_top_ks:
                    key = str(top_k)
                    top_k_tokens = (sample_tokens.get("top_k_tokens") or {}).get(key)
                    if top_k_tokens:
                        rebuilt_tokens_by_k[key].append({"sample_id": sample_id, **top_k_tokens})
        return rebuilt_results, rebuilt_tokens, rebuilt_tokens_by_k

    def handle_worker_result(completed):
        nonlocal results_by_k, token_stats, token_stats_by_k
        completed_by_index[completed["dataset_index"]] = completed
        results_by_k, token_stats, token_stats_by_k = rebuild_aggregates()
        for top_k in retrieve_top_ks:
            key = str(top_k)
            write_json(top_k_outputs[key], results_by_k[key])
        max_key = str(max(int(k) for k in retrieve_top_ks))
        write_json(output_path, results_by_k[max_key])
        processed_count = len(completed_by_index)
        action = "Cached" if completed.get("cached") else "Completed"
        print(f"[{processed_count}/{len(selected_samples)}] {action} {execution_pipeline}: {completed['sample_id']}", flush=True)
        write_status(
            status_path,
            {
                "status": "running",
                "processed_samples": processed_count,
                "current_sample_id": completed["sample_id"],
                **status_common,
            },
        )

    try:
        if num_processes == 1:
            for task in tasks:
                handle_worker_result(process_sample_worker(task))
        else:
            start_method = str(extras.get("multiprocessing_start_method", "spawn"))
            context = multiprocessing.get_context(start_method)
            print(
                f"Starting LightMem sample pool: processes={num_processes}, start_method={start_method}",
                flush=True,
            )
            with context.Pool(
                processes=num_processes,
                initializer=initialize_sample_worker,
                initargs=(worker_cuda_device_count,),
            ) as pool:
                for completed in pool.imap_unordered(process_sample_worker, tasks, chunksize=1):
                    handle_worker_result(completed)
    except Exception as exc:
        write_status(
            status_path,
            {
                "status": "failed",
                "processed_samples": len(completed_by_index),
                "error": str(exc),
                **status_common,
            },
        )
        raise

    if track_tokens:
        aggregate_token_stats(token_file, f"lightmem-{dataset_type}-official", token_stats)
        for top_k in retrieve_top_ks:
            key = str(top_k)
            aggregate_token_stats(
                top_k_token_path(output_dir, int(top_k)),
                f"lightmem-{dataset_type}-official-top-k-{top_k}",
                token_stats_by_k[key],
            )

    run_summary = {
        "dataset_type": dataset_type,
        "execution_pipeline": execution_pipeline,
        "retrieve_top_ks": retrieve_top_ks,
        "output_dir": output_dir,
        "top_k_outputs": top_k_outputs,
        "token_file": token_file,
        "status_file": status_path,
        "samples": len(selected_samples),
        "num_processes": num_processes,
        "worker_cuda_device_count": worker_cuda_device_count,
        "llmlingua_service_url": llmlingua_service_url or None,
        "track_tokens": track_tokens,
    }
    write_json(os.path.join(output_dir, "run_summary.json"), run_summary)

    if dataset_type == "locomo" and bool(extras.get("auto_eval", False)):
        eval_dataset = extras.get("eval_dataset", "loco")
        result_dataset_dir = "LONGMEMEVAL" if eval_dataset == "lme" else "LOCOMO"
        result_root = os.path.join(CODE_DIR, "Result", result_dataset_dir, "lightmem")
        try:
            version_base = os.path.relpath(output_dir, result_root)
        except ValueError:
            version_base = os.path.basename(output_dir.rstrip(os.sep))
        if version_base.startswith(".."):
            version_base = os.path.basename(output_dir.rstrip(os.sep))

        from eval import main as eval_main

        eval_method = extras.get("eval_method", "lightmem")
        eval_embedding_model = extras.get("eval_embedding_model") or params["embedding_model_name"]
        eval_result_path = extras.get("eval_result_path")
        if not eval_result_path:
            raise ValueError(
                "LightMem auto_eval requires eval_result_path pointing to a separately "
                "simplified result file; raw generation is saved as result_raw.json"
            )
        for top_k in retrieve_top_ks:
            version = os.path.join(version_base, f"top_k_{top_k}")
            print(f"Running evaluation for lightmem top_k={top_k}, version={version}")
            result_path = str(eval_result_path).format(top_k=top_k)
            eval_main(
                eval_dataset,
                eval_method,
                version,
                eval_embedding_model,
                result_path=result_path,
            )

    write_status(
        status_path,
        {"status": "complete", "processed_samples": len(selected_samples), **status_common},
    )

    if dataset_type == "locomo":
        return results_by_k
    return results_by_k[str(params["retrieve_k"])]
