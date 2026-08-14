import time
import uuid
import os
import openai
import numpy as np
from openai import OpenAI
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

DEFAULT_LLM_API_KEY = "empty"
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_LLM_MODEL = "Qwen3.5-9B"
DEFAULT_EMBEDDING_MODEL_NAME = "/path/to/local/all-MiniLM-L6-v2"

_RUNTIME_CONFIG = {
    "llm_api_key": DEFAULT_LLM_API_KEY,
    "llm_base_url": DEFAULT_LLM_BASE_URL,
    "llm_model": DEFAULT_LLM_MODEL,
    "embedding_model_name": DEFAULT_EMBEDDING_MODEL_NAME,
    "embedding_api_key": None,
    "embedding_base_url": None,
}
_EMBEDDING_MODEL_CACHE = {}
_EMBEDDING_CLIENT_CACHE = {}
_TOKENIZER_CACHE = {}
DEFAULT_CONTEXT_LENGTH = 20000
INPUT_TOKEN_RESERVE = 128
TOKENIZER_MODEL_PATHS = {
    "Qwen3.5-9B": "/path/to/local/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B": "/path/to/local/Qwen3.5-9B",
    "Qwen3.5-27B": "/path/to/local/Qwen3.5-27B",
    "Qwen/Qwen3.5-27B": "/path/to/local/Qwen3.5-27B",
}


def configure_memoryos_runtime(
    llm_api_key=None,
    llm_base_url=None,
    llm_model=None,
    embedding_model_name=None,
    embedding_api_key=None,
    embedding_base_url=None,
):
    if llm_api_key:
        _RUNTIME_CONFIG["llm_api_key"] = llm_api_key
    if llm_base_url:
        _RUNTIME_CONFIG["llm_base_url"] = llm_base_url
    if llm_model:
        _RUNTIME_CONFIG["llm_model"] = llm_model
    if embedding_model_name:
        _RUNTIME_CONFIG["embedding_model_name"] = embedding_model_name

    if embedding_api_key is not None:
        _RUNTIME_CONFIG["embedding_api_key"] = embedding_api_key
    if embedding_base_url is not None:
        _RUNTIME_CONFIG["embedding_base_url"] = embedding_base_url

def get_llm_model():
    return _RUNTIME_CONFIG["llm_model"]


def get_embedding_model_name():
    return _RUNTIME_CONFIG["embedding_model_name"]


def build_default_client():
    return OpenAIClient(
        api_key=_RUNTIME_CONFIG["llm_api_key"],
        base_url=_RUNTIME_CONFIG["llm_base_url"],
    )


def resolve_tokenizer_name(model_name):
    return TOKENIZER_MODEL_PATHS.get(model_name, model_name)


def uses_responses_api(model_name):
    return str(model_name).lower().startswith("gpt-5.4")


def get_chat_tokenizer(model_name):
    if str(model_name).lower().startswith("deepseek-") or uses_responses_api(model_name):
        return None
    if AutoTokenizer is None:
        return None
    tokenizer_name = resolve_tokenizer_name(model_name)
    tokenizer = _TOKENIZER_CACHE.get(tokenizer_name)
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        except Exception as exc:
            print(f"Warning: failed to load tokenizer for {model_name}: {exc}")
            tokenizer = None
        _TOKENIZER_CACHE[tokenizer_name] = tokenizer
    return tokenizer


def _copy_messages(messages):
    return [dict(message) for message in messages]


def _estimate_message_tokens(messages):
    # Conservative fallback when the real tokenizer is unavailable.
    total = 0
    for message in messages:
        content = str(message.get("content", ""))
        total += max(1, len(content) // 3) + 8
    return total + 16


def _count_message_tokens(messages, tokenizer, extra_body=None):
    if tokenizer is None:
        return _estimate_message_tokens(messages)
    try:
        chat_kwargs = {}
        if isinstance(extra_body, dict):
            chat_kwargs = extra_body.get("chat_template_kwargs") or {}
        return len(tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **chat_kwargs,
        ))
    except Exception:
        try:
            text = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
            return len(tokenizer.encode(text)) + 16
        except Exception:
            return _estimate_message_tokens(messages)


def _truncate_longest_user_message(messages, tokenizer, max_input_tokens, extra_body=None):
    token_count = _count_message_tokens(messages, tokenizer, extra_body)
    if token_count <= max_input_tokens:
        return messages, token_count, False

    truncated = _copy_messages(messages)
    candidate_indices = [
        idx for idx, message in enumerate(truncated)
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    ]
    if not candidate_indices:
        candidate_indices = [
            idx for idx, message in enumerate(truncated)
            if isinstance(message.get("content"), str)
        ]
    if not candidate_indices:
        return truncated, token_count, False

    target_idx = max(candidate_indices, key=lambda idx: len(truncated[idx].get("content", "")))
    original_content = truncated[target_idx].get("content", "")
    marker = "\n\n[Input truncated to fit the model context window.]"

    lo, hi = 0, len(original_content)
    best_content = original_content[:0] + marker
    best_count = _count_message_tokens(truncated, tokenizer, extra_body)
    while lo <= hi:
        mid = (lo + hi) // 2
        truncated[target_idx]["content"] = original_content[:mid] + marker
        current_count = _count_message_tokens(truncated, tokenizer, extra_body)
        if current_count <= max_input_tokens:
            best_content = truncated[target_idx]["content"]
            best_count = current_count
            lo = mid + 1
        else:
            hi = mid - 1

    truncated[target_idx]["content"] = best_content
    print(
        f"Truncated LLM input from {token_count} to {best_count} tokens "
        f"for max_tokens={max_input_tokens}."
    )
    return truncated, best_count, True


def truncate_messages_for_context(model, messages, max_tokens, extra_body=None, reserve_tokens=INPUT_TOKEN_RESERVE):
    context_length = (
        32768
        if str(model).lower().startswith("deepseek-") or uses_responses_api(model)
        else DEFAULT_CONTEXT_LENGTH
    )
    max_input_tokens = max(1, context_length - int(max_tokens or 0) - int(reserve_tokens or 0))
    tokenizer = get_chat_tokenizer(model)
    return _truncate_longest_user_message(_copy_messages(messages), tokenizer, max_input_tokens, extra_body)[0]


def get_embedding_model(model_name=None):
    model_name = model_name or get_embedding_model_name()
    device = os.environ.get("MEMORYOS_EMBEDDING_DEVICE", "cpu")
    cache_key = (model_name, device)
    model = _EMBEDDING_MODEL_CACHE.get(cache_key)
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "Local embedding requires a working sentence-transformers installation"
            ) from exc
        model = SentenceTransformer(model_name, device=device)
        _EMBEDDING_MODEL_CACHE[cache_key] = model
        print(f"Embedding model loaded on {device}: {model_name}")
    return model


def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def generate_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def get_embedding(text, model_name=None):
    model_name = model_name or get_embedding_model_name()
    embedding_base_url = _RUNTIME_CONFIG.get("embedding_base_url")
    if not embedding_base_url:
        model = get_embedding_model(model_name)
        return model.encode([text], convert_to_numpy=True)[0]

    api_key = _RUNTIME_CONFIG.get("embedding_api_key") or "EMPTY"
    cache_key = (api_key, embedding_base_url)
    client = _EMBEDDING_CLIENT_CACHE.get(cache_key)
    if client is None:
        client = OpenAI(
            api_key=api_key,
            base_url=embedding_base_url,
            max_retries=5,
            timeout=float(os.environ.get("MEMORYOS_EMBEDDING_TIMEOUT_SECONDS", "120")),
        )
        _EMBEDDING_CLIENT_CACHE[cache_key] = client
        print(f"Remote embedding service configured: {embedding_base_url}")
    response = client.embeddings.create(
        model=model_name,
        input=str(text),
        extra_body={"truncate_prompt_tokens": 256},
    )
    return np.asarray(response.data[0].embedding, dtype=np.float32)

def normalize_vector(vec):
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

class OpenAIClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        openai.api_key = self.api_key
        openai.api_base = self.base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=5,
            timeout=float(os.environ.get("MEMORYOS_LLM_TIMEOUT_SECONDS", "120")),
        )

    def chat_completion(self, model, messages, temperature=0.0, max_tokens=2000, extra_body=None):
        print("Calling chat completion with model:", model)
        if extra_body is None and "qwen" in str(model).lower():
            extra_body = {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        prepared_messages = truncate_messages_for_context(model, messages, max_tokens, extra_body)
        last_error = None
        for attempt in range(5):
            try:
                if uses_responses_api(model):
                    output_budget = min(int(max_tokens) * (2 ** attempt), 2000)
                    response = self.client.responses.create(
                        model=model,
                        input=prepared_messages,
                        max_output_tokens=output_budget,
                    )
                    content = response.output_text
                else:
                    request = {
                        "model": model,
                        "messages": prepared_messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if extra_body is not None:
                        request["extra_body"] = extra_body
                    response = self.client.chat.completions.create(**request)
                    content = response.choices[0].message.content
                if isinstance(content, str) and content.strip():
                    return content.strip()
                last_error = RuntimeError(
                    f"LLM returned empty content for model {model} on attempt {attempt + 1}/5"
                )
                if attempt < 4:
                    print(f"Warning: {last_error}; retrying.")
                    time.sleep(min(2 ** attempt, 8))
            except Exception as exc:
                message = str(exc)
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                retryable = (
                    isinstance(
                        exc,
                        (
                            openai.APITimeoutError,
                            openai.APIConnectionError,
                            openai.RateLimitError,
                            openai.InternalServerError,
                        ),
                    )
                    or status_code in (408, 409, 429)
                    or (isinstance(status_code, int) and status_code >= 500)
                )
                if retryable:
                    if attempt < 4:
                        delay = min(2 ** attempt, 8)
                        print(
                            f"Warning: transient LLM error on attempt {attempt + 1}/5: "
                            f"{type(exc).__name__}: {exc}; retrying in {delay}s."
                        )
                        time.sleep(delay)
                        continue
                    break
                if "context length" not in message and "maximum input length" not in message and "input tokens" not in message:
                    raise
                prepared_messages = truncate_messages_for_context(
                    model,
                    prepared_messages,
                    max_tokens,
                    extra_body,
                    reserve_tokens=INPUT_TOKEN_RESERVE * (attempt + 2),
                )
        raise last_error

def gpt_generate_answer(prompt, messages, client, model=None):
    return client.chat_completion(
        model=model or get_llm_model(),
        messages=messages,
        temperature=0.0,
        max_tokens=2000,
    )

def analyze_assistant_knowledge(dialogs, client):
    """
    Analyzes conversations to extract knowledge or identity traits about the assistant.
    Returns: {"assistant_knowledge": str}
    """
    conversation = "\n".join([f"User: {d['user_input']}\nAI: {d['agent_response']}\nTime:{d['timestamp']}\n" for d in dialogs])

    prompt = """
# Assistant Knowledge Extraction Task
Analyze the conversation and extract any fact or identity traits about the assistant. 
If no traits can be extracted, reply with "None". Use the following format for output:
The generated content should be as concise as possible — the more concise, the better.
【Assistant Knowledge】
 [Fact 1]
 [Fact 2]
 (Or "None" if none found)

Few-shot examples:
1. User: Can you recommend some movies.
   AI: Yes, I recommend Interstellar.
   Time: 2023-10-01
   【Assistant Knowledge】
   - I recommend Interstellar on 2023-10-01.

2. User: Can you help me with cooking recipes?
   AI: Yes, I have extensive knowledge of cooking recipes and techniques.
   Time: 2023-10-02
   【Assistant Knowledge】
   - I have cooking recipes and techniques on 2023-10-02.

3. User: That’s interesting. I didn’t know you could do that.
   AI: I’m glad you find it interesting!
   【Assistant Knowledge】
   - None

Conversation:
""" + conversation

    messages = [
        {
            "role": "system",
            "content": """You are an assistant knowledge extraction engine. Rules:
1. Extract ONLY explicit statements about the assistant's identity or knowledge.
2. Use concise and factual statements in the first person.
3. If no relevant information is found, output "None".""" 
        },
        {"role": "user", "content": prompt}
    ]

    print("Analyzing assistant knowledge...")
    result = gpt_generate_answer(prompt, messages, client)
    
    # Parse output
    assistant_knowledge = result.replace("【Assistant Knowledge】", "").strip()
    return {"assistant_knowledge": assistant_knowledge}

def gpt_summarize(dialogs, client):
    prompt = "Please generate a topic summary based on the following conversation：\n"
    for d in dialogs:
        prompt += f"user: {d.get('user_input','')}\nassiant: {d.get('agent_response','')}\n"
    prompt += "\nSubject Summary："
    messages = [
        {"role": "system", "content": "You are an expert in summarizing dialogue topics, please generate a concise and precise summary."},
        {"role": "user", "content": prompt}
    ]
    print("Calling GPT to generate a topic summary...")
    return gpt_generate_answer(prompt, messages, client)

def normalize_multi_summaries(parsed):
    if isinstance(parsed, dict):
        nested = parsed.get("summaries")
        if isinstance(nested, list):
            parsed = nested
        elif any(key in parsed for key in ("theme", "keywords", "content")):
            parsed = [parsed]
        else:
            parsed = []
    elif isinstance(parsed, str):
        parsed = [parsed]
    elif not isinstance(parsed, list):
        parsed = []

    normalized = []
    for item in parsed:
        if isinstance(item, str):
            content = item.strip()
            if content:
                normalized.append({"theme": "", "keywords": [], "content": content})
            continue
        if not isinstance(item, dict):
            continue

        theme = str(item.get("theme", "") or "").strip()
        content = str(item.get("content", "") or "").strip()
        keywords = item.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [part.strip() for part in keywords.split(",") if part.strip()]
        elif isinstance(keywords, list):
            keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
        else:
            keywords = []
        if theme or content:
            normalized.append({"theme": theme, "keywords": keywords, "content": content})
    return normalized

def gpt_generate_multi_summary(text, client):
    """
    Call the LLM to generate multiple subtopic summaries.
    Example return format:
    {
      "input": "dialog text",
      "summaries": [
         {"theme": "Business trip", "keywords": ["Business trip", "Itinerary", "Work"], "content": "The user mentioned difficulties related to business trips."},
         {"theme": "Health", "keywords": ["Cold", "Uncomfortable", "Sick"], "content": "The user reported discomfort caused by a cold."}
      ]
    }
    """
    prompt = ("Please analyze the following dialogue and generate multiple subtopic summaries (if applicable), with a maximum of two themes.\n"
              "Each summary should include the subtopic name, keywords (separated by commas), and the summary text, formatted as a JSON array, with an example format as follows:\n"
              "[\n  {\"theme\": \"Business trip\", \"keywords\": [\"Business trip\", \"Itinerary\", \"Work\"], \"content\": \" User mentioned the troubles related to business trips.\"},\n  {\"theme\": \"Health\", \"keywords\": [\"Cold\", \"Uncomfortable\", \"Sick\"], \"content\": \"User reported feeling unwell due to a cold.\"}\n]\n"
              "Please directly output the JSON array, without adding any other content.\nConversation content:\n" + text)
    messages = [
        {"role": "system", "content": "You are an expert in analyzing dialogue topics. No more than two topics."},
        {"role": "user", "content": prompt}
    ]
    print("Calling GPT to generate multi-topic summaries...")
    response_text = gpt_generate_answer(prompt, messages, client)
    import json
    try:
        summaries = normalize_multi_summaries(json.loads(response_text))
    except Exception:
        summaries = []
    return {"input": text, "summaries": summaries}

# def gpt_personality_analysis(dialogs, client):
#     prompt = ("Please analyze the following conversation and extract the user profile information and user private data."
#               "Please output in the following format:\n"
#               "【User Profile】\n"
#               "Areas of Interest:\n"
#               "Response Preferences：\n"
#               "Preferred Content Type：\n"
#               "Short vs. Detailed Responses：\n"
#               "Formal vs. Casual Tone：\n"
#               "Other Notes:：\n"
#               "【User Private Data】\n"
#               "Please list all the private information involved (such as account numbers, passwords, user purchase,etc.). If there is none, please write \"None\"\n\n"
#               "The conversation is as follows:\n")
#     for d in dialogs:
#         prompt += f"User: {d.get('user_input','')}\nAssiant: {d.get('agent_response','')}\n"
#     messages = [
#         {"role": "system", "content": "You are a professional user profile analyst who can also identify user private data. Please strictly follow the template for output."},
#         {"role": "user", "content": prompt}
#     ]
#     print("Calling GPT to analyze the user profile and private data...")
#     result_text = gpt_generate_answer(prompt, messages, client)
#     profile, private = "", ""
#     parts = result_text.split("【User Private Data】")
#     if len(parts) == 2:
#         profile = parts[0].replace("【User Profile】", "").strip()
#         private = parts[1].strip()
#     else:
#         profile = result_text.strip()
#         private = "None"
#     return {"profile": profile, "private": private}
# def gpt_personality_analysis(dialogs, client):
#     """
#     Analyzes conversations to extract structured personality traits, private knowledge, 
#     and assistant-related knowledge.
#     Returns: {"profile": str, "private": str, "assistant_knowledge": str}
#     """
#     conversation = "\n".join([f"User: {d['user_input']}\nAssistant: {d['agent_response']}" for d in dialogs])

#     prompt = """
# # Personality Analysis Task
# Analyze the conversation and output in EXACTLY this format:

# 【User Profile】
# 1. Core Psychological Traits:
#    - [Trait]: [Positive/Negative/Neutral] (Evidence)
#    - (Max 5 most prominent traits)

# 2. Content Preferences:
#    - [Topic]: [Like/Dislike/Neutral] (Evidence)
#    - (Max 5 strongest preferences)

# 3. Interaction Style:
#    - [Style]: [Preference] (Evidence)
#    - (e.g., Direct/Indirect, Detailed/Concise)

# 4. Value Alignment:
#    - [Value]: [Strong/Weak] (Evidence)
#    - (e.g., Honesty, Helpfulness)

# 【User Private Data】
# - [Fact 1]
# - [Fact 2]
# - (Or "None" if none found)

# Conversation:
# """ + conversation

#     messages = [
#         {
#             "role": "system",
#             "content": """You are a personality analysis engine. Rules:
# 1. Extract ONLY observable traits with direct evidence
# 2. Use standardized trait names from psychology
# 3. Mark confidence: Positive=explicit preference, Neutral=implied
# 4. Private data includes possessions, habits, and sensitive preferences"""
#         },
#         {"role": "user", "content": prompt}
#     ]

#     print("Running personality analysis...")
#     result = gpt_generate_answer(prompt, messages, client)
    
#     # Parse output
#     profile, private = result.split("【User Private Data】") if "【User Private Data】" in result else (result, "None")
    
#     # Analyze assistant knowledge
#     assistant_knowledge_result = analyze_assistant_knowledge(dialogs, client)
    
#     return {
#         "profile": profile.replace("【User Profile】", "").strip(),
#         "private": private.strip(),
#         "assistant_knowledge": assistant_knowledge_result["assistant_knowledge"]
#     }
def gpt_personality_analysis(dialogs, client):
    """
    Analyzes conversations to extract structured personality traits, general user data, 
    and assistant-related knowledge.
    Returns: {"profile": str, "user_data": str, "assistant_knowledge": str}
    """
    conversation = "\n".join([f"User: {d['user_input']}\nAssistant: {d['agent_response']}\nTime:{d['timestamp']}" for d in dialogs])

    prompt = """
# Personality and User Data Analysis Task
Analyze the conversation and output in EXACTLY this format:

【User Profile】
1. Core Psychological Traits:
   - [Trait]: [Positive/Negative/Neutral] (Evidence)
   - (Max 5 most prominent traits)

2. Content Preferences:
   - [Topic]: [Like/Dislike/Neutral] (Evidence)
   - (Max 5 strongest preferences)

3. Interaction Style:
   - [Style]: [Preference] (Evidence)
   - (e.g., Direct/Indirect, Detailed/Concise)

4. Value Alignment:
   - [Value]: [Strong/Weak] (Evidence)
   - (e.g., Honesty, Helpfulness)

【User Data】
 [Fact 1]: [Details] (e.g., "User mentioned visiting a park on April 1st, 2025 in New York.")
 [Fact 2]: [Details] (e.g., "User likes pizza, enjoys sci-fi movies, and dislikes rainy weather.")
 (Include events, dates, locations, preferences, or other general or private information explicitly mentioned in the conversation. If none, write "None.")

Conversation:
""" + conversation
    messages = [
        {
            "role": "system",
            "content": """You are a personality and user data analysis engine. Rules:
1. Extract ONLY observable traits and data with direct evidence.
2. Include general user data such as events, dates, locations, and preferences.
3. Use concise and factual statements.
4. If no relevant information is found, output "None"."""
        },
        {"role": "user", "content": prompt}
    ]

    print("Running personality and user data analysis...")
    result = gpt_generate_answer(prompt, messages, client)
    
    # Parse output
    profile, user_data = result.split("【User Data】") if "【User Data】" in result else (result, "None")
    
    # Analyze assistant knowledge
    assistant_knowledge_result = analyze_assistant_knowledge(dialogs, client)
    
    return {
        "profile": profile.replace("【User Profile】", "").strip(),
        "private": user_data.strip(),
        "assistant_knowledge": assistant_knowledge_result["assistant_knowledge"]
    }

def gpt_update_profile(old_profile, new_analysis, client):
    """
    Dynamically merges old and new profile data
    Args:
        old_profile: Previous profile text (structured)
        new_analysis: New analysis text (same format)
    Returns:
        Merged profile text with conflict resolution
    """
    prompt = f"""
# Profile Merge Task
Consolidate these profiles while:
 Preserving all valid observations
 Resolving conflicts
 Adding new dimensions

## Current Profile
{old_profile}

## New Data
{new_analysis}

## Rules
1. Keep ALL verified traits from both
2. Resolve conflicts by:
   a) New explicit evidence > old assumptions
   b) Mark as Neutral if contradictory
3. Add new dimensions from new data
4. Maintain EXACT original format

Output ONLY the merged profile (no commentary):
The generated content should not exceed 1500 words
"""

    messages = [
        {
            "role": "system",
            "content": """You are a profile integration system. Your rules:
1. NEVER discard verified information
2. Conflict resolution hierarchy:
   Explicit statement > Implied trait > Assumption
3. Add timestamps when traits change:
   (Updated: [date]) for modified traits
4. Preserve the 4-category structure"""
        },
        {"role": "user", "content": prompt}
    ]

    print("Updating user profile dynamically...")
    return gpt_generate_answer(prompt, messages, client)

def gpt_extract_theme(answer_text, client):
    prompt = (
        "Please extract a concise topic summary from the following answer and prefix the output with "
        '"【Topic Extraction】:"\n'
        f"{answer_text}\n"
    )
    messages = [
        {"role": "system", "content": "You are an expert in extracting conversation topics."},
        {"role": "user", "content": prompt}
    ]
    print("Calling GPT to extract the topic summary...")
    return gpt_generate_answer(prompt, messages, client)

def llm_extract_keywords(text, client):
    prompt = "Please extract the keywords of the conversation topic from the following dialogue, separated by commas, and do not exceed three:\n" + text
    messages = [
        {"role": "system", "content": "You are a keyword extraction expert. Please extract the keywords of the conversation topic."},
        {"role": "user", "content": prompt}
    ]
    print("Calling GPT to extract keywords...")
    keywords_text =gpt_generate_answer(prompt, messages, client)
    keywords = [w.strip() for w in keywords_text.split(",") if w.strip()]
    return set(keywords)

def compute_time_decay(session_timestamp, current_timestamp, tau=3600):
    from datetime import datetime
    fmt = "%Y-%m-%d %H:%M:%S"
    t1 = datetime.strptime(session_timestamp, fmt)
    t2 = datetime.strptime(current_timestamp, fmt)
    delta = (t2 - t1).total_seconds()
    return np.exp(-delta/tau)
