import json
import os
import re
import time
from contextlib import contextmanager, nullcontext

import requests
from openai import OpenAI

DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
DEFAULT_LLM_API_KEY = "empty"
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_API_KEY = "empty"
DEFAULT_EMBEDDING_BASE_URL = "http://localhost:7999/v1"
DEFAULT_EMBEDDING_ENDPOINT_TYPE = "openai"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_LETTA_BASE_URL = "http://localhost:8283"
DEFAULT_RETRIEVE_K = 10


class StageNode:
    def __init__(self, name):
        self.name = name
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.start_time = None
        self.end_time = None
        self.duration_seconds = 0.0
        self.sub_stages = {}

    def to_dict(self):
        return {
            "name": self.name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "sub_stages": {key: value.to_dict() for key, value in self.sub_stages.items()},
        }


class TokenTracker:
    def __init__(self, output_file):
        self.output_file = output_file
        self.root = StageNode("root")
        self.stack = [self.root]

    @contextmanager
    def stage(self, name):
        parent = self.stack[-1]
        if name in parent.sub_stages:
            node = parent.sub_stages[name]
        else:
            node = StageNode(name)
            parent.sub_stages[name] = node
        node.start_time = time.time()
        self.stack.append(node)
        try:
            yield node
        finally:
            node.end_time = time.time()
            node.duration_seconds += node.end_time - node.start_time
            self.stack.pop()

    def add_tokens(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        for node in self.stack:
            node.prompt_tokens += int(prompt_tokens)
            node.completion_tokens += int(completion_tokens)
            node.total_tokens += int(total_tokens)

    def save_to_json(self):
        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as file_obj:
            json.dump(self.root.to_dict(), file_obj, ensure_ascii=False, indent=2)


def import_letta_client():
    try:
        from letta_client import Letta
    except ImportError as exc:
        raise ImportError(
            "MemGPT requires the official `letta_client` package in the current environment."
        ) from exc

    try:
        from letta_client.types import EmbeddingConfig, LlmConfig
    except ImportError:
        try:
            from letta_client.types import EmbeddingConfig, LLMConfig as LlmConfig
        except ImportError as exc:
            raise ImportError(
                "Could not import Letta client config types. Please install a compatible `letta_client` version."
            ) from exc

    return Letta, LlmConfig, EmbeddingConfig


def normalize_url(url):
    return (url or "").rstrip("/")


def build_openai_client(base_url, api_key):
    return OpenAI(base_url=normalize_url(base_url), api_key=api_key or DEFAULT_LLM_API_KEY)


def extract_first_sentence(text):
    if not text:
        return ""
    normalized = re.sub(r"^(Answer|A):\s*", "", str(text).strip(), flags=re.IGNORECASE)
    parts = re.split(r"[。.!?\n]", normalized, maxsplit=1)
    return parts[0].strip()


def is_bad_answer(answer):
    if not answer:
        return True
    lowered = answer.lower()
    bad_answers = ["i don't know", "not sure", "unknown", "no information", "none"]
    return any(item in lowered for item in bad_answers)


def extract_usage(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def call_llm(messages, client, model_name, tracker=None, stage_name=None, temperature=0.0, max_tokens=200):
    stage_cm = tracker.stage(stage_name) if tracker is not None and stage_name else nullcontext()
    with stage_cm:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = extract_usage(response)
        if tracker is not None:
            tracker.add_tokens(**usage)
        return response.choices[0].message.content.strip()


def parse_archival_items(payload):
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("items") or payload.get("data") or []
    else:
        items = []

    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if text:
            results.append(str(text))
    return results


def search_archival(agent_id, query, letta_base_url, retrieve_k):
    response = requests.get(
        f"{normalize_url(letta_base_url)}/v1/agents/{agent_id}/archival-memory",
        params={"search": query, "limit": retrieve_k},
        timeout=60,
    )
    response.raise_for_status()
    return parse_archival_items(response.json())[:retrieve_k]


def search_recall(question, recall_file, retrieve_k):
    if not os.path.exists(recall_file):
        return []

    with open(recall_file, "r", encoding="utf-8") as file_obj:
        memories = json.load(file_obj)

    keywords = [word.lower() for word in re.findall(r"\w+", question) if len(word) > 2]
    matched = []
    for memory in memories:
        text = str(memory)
        lowered = text.lower()
        if any(keyword in lowered for keyword in keywords):
            matched.append(text)
        if len(matched) >= retrieve_k:
            break
    return matched


def write_recall_memory(recall_file, content):
    memories = []
    if os.path.exists(recall_file):
        with open(recall_file, "r", encoding="utf-8") as file_obj:
            memories = json.load(file_obj)
    memories.append(content)
    with open(recall_file, "w", encoding="utf-8") as file_obj:
        json.dump(memories, file_obj, ensure_ascii=False, indent=2)


def answer_question(question, retrieved_texts, client, model_name, tracker=None, stage_name=None):
    context_text = "\n\n".join(retrieved_texts) if retrieved_texts else "No specific memory found."
    prompt = f"""
You are an intelligent memory assistant tasked with retrieving accurate information from the provided conversation memories.

# CONTEXT:
{context_text}

# INSTRUCTIONS:
1. Analyze the context carefully.
2. Focus on explicit facts, dates, events, and entities.
3. Pay attention to timestamps if present.
4. Use only information from the context; do not guess.
5. Provide a concise answer: 5 words or less.
6. If unsure, respond "unknown".

# QUESTION:
{question}

# ANSWER:
"""
    return call_llm(
        [{"role": "user", "content": prompt}],
        client,
        model_name,
        tracker=tracker,
        stage_name=stage_name,
        temperature=0.0,
        max_tokens=128,
    )


def agent_answer(
    agent_id,
    question,
    llm_client,
    llm_model,
    letta_base_url,
    recall_file,
    retrieve_k,
    tracker=None,
    qa_idx=None,
):
    decision_prompt = f"""
You are an intelligent conversation agent.

Your goal: Decide whether answering this question requires:
1. searching the archival memory,
2. searching the recall memory, or
3. answering directly from your knowledge.

Question: {question}

# RULES:
- Prefer recall memory if the fact may have been previously stored during this QA process.
- Use archival memory if it involves events or conversations in the dataset.
- Answer directly only if no memory is needed.
- Respond with ONE of: search_archival / search_recall / direct
"""

    decision = call_llm(
        [{"role": "user", "content": decision_prompt}],
        llm_client,
        llm_model,
        tracker=tracker,
        stage_name=f"QA{qa_idx}-decision" if qa_idx is not None else None,
        temperature=0.0,
        max_tokens=16,
    )

    tool = "direct"
    if "search_archival" in decision.lower():
        tool = "search_archival"
    elif "search_recall" in decision.lower():
        tool = "search_recall"

    retrieved = []
    if tool == "search_archival":
        retrieved = search_archival(agent_id, question, letta_base_url, retrieve_k)
    elif tool == "search_recall":
        retrieved = search_recall(question, recall_file, retrieve_k)

    answer = answer_question(
        question,
        retrieved,
        llm_client,
        llm_model,
        tracker=tracker,
        stage_name=f"QA{qa_idx}-answer" if qa_idx is not None else None,
    )

    if is_bad_answer(answer) and tool != "search_archival":
        fallback_retrieved = search_archival(agent_id, question, letta_base_url, retrieve_k)
        fallback_answer = answer_question(
            question,
            fallback_retrieved,
            llm_client,
            llm_model,
            tracker=tracker,
            stage_name=f"QA{qa_idx}-fallback" if qa_idx is not None else None,
        )
        if fallback_retrieved:
            retrieved = fallback_retrieved
        answer = fallback_answer

    memory_prompt = f"""
You are a memory curator.

# INPUT:
Q: {question}
A: {answer}

# RULES TO WRITE RECALL MEMORY:
1. Extract ONE clear, important fact from the QA.
2. Only store timestamped events, facts, or entities useful for future conversations.
3. Ignore trivial conversational text.
4. Format as a single concise sentence.
5. If no important fact is present, output NONE.

# OUTPUT:
"""
    memory_fact = call_llm(
        [{"role": "user", "content": memory_prompt}],
        llm_client,
        llm_model,
        tracker=tracker,
        stage_name=f"QA{qa_idx}-memory_write" if qa_idx is not None else None,
        temperature=0.0,
        max_tokens=64,
    )

    if memory_fact and "none" not in memory_fact.lower() and len(memory_fact) < 200:
        write_recall_memory(recall_file, memory_fact.strip())

    return extract_first_sentence(answer), retrieved


def write_json(path, data):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


def load_json_if_exists(path, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def safe_name(value):
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return normalized.strip("._") or "sample"


def build_cache_file(cache_dir, index, sample_id):
    return os.path.join(cache_dir, f"{index:05d}_{safe_name(sample_id)}.json")


def collect_existing_results(output_path, subset_samples, cache_dir):
    results_by_id = {}

    existing_output = load_json_if_exists(output_path, [])
    if isinstance(existing_output, list):
        for sample in existing_output:
            sample_id = sample.get("sample_id")
            if sample_id is not None:
                results_by_id[str(sample_id)] = sample

    for index, sample in enumerate(subset_samples):
        sample_id = str(sample.get("sample_id", f"sample_{index}"))
        cache_file = build_cache_file(cache_dir, index, sample_id)
        if os.path.exists(cache_file):
            results_by_id[sample_id] = load_json_if_exists(cache_file, {})

    return results_by_id


def aggregate_results(output_path, subset_samples, cache_dir, results_by_id):
    aggregated = []
    for index, sample in enumerate(subset_samples):
        sample_id = str(sample.get("sample_id", f"sample_{index}"))
        cache_file = build_cache_file(cache_dir, index, sample_id)
        sample_result = None
        if os.path.exists(cache_file):
            sample_result = load_json_if_exists(cache_file, {})
            results_by_id[sample_id] = sample_result
        elif sample_id in results_by_id:
            sample_result = results_by_id[sample_id]

        if sample_result:
            aggregated.append(sample_result)

    write_json(output_path, aggregated)
    return aggregated


def normalize_conversation_entry(message):
    text = str(message.get("text", ""))
    if message.get("blip_caption"):
        text = f"{text} (image description: {message['blip_caption']})"
    return text


def load_conversation_into_archival_memory(agent_id, sample, letta_base_url):
    conversation = sample.get("conversation", {})
    for key in sorted(conversation):
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue
        timestamp = conversation.get(f"{key}_date_time", "N/A")
        lines = [f"--- Session Date: {timestamp} ---"]
        for message in conversation.get(key, []):
            speaker = message.get("speaker", "Unknown")
            lines.append(f"{speaker}: {normalize_conversation_entry(message)}")

        response = requests.post(
            f"{normalize_url(letta_base_url)}/v1/agents/{agent_id}/archival-memory",
            json={"text": "\n".join(lines)},
            timeout=120,
        )
        response.raise_for_status()


def run_memgpt_workflow(
    dataset_path,
    output_path,
    token_file,
    llm_model=DEFAULT_LLM_MODEL,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_api_key=DEFAULT_EMBEDDING_API_KEY,
    embedding_base_url=DEFAULT_EMBEDDING_BASE_URL,
    embedding_endpoint_type=DEFAULT_EMBEDDING_ENDPOINT_TYPE,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    letta_base_url=DEFAULT_LETTA_BASE_URL,
    retrieve_k=DEFAULT_RETRIEVE_K,
    start_idx=0,
    end_idx=None,
):
    Letta, LlmConfig, EmbeddingConfig = import_letta_client()

    output_dir = os.path.dirname(output_path)
    cache_dir = os.path.join(output_dir, "_memgpt_cache")
    os.makedirs(cache_dir, exist_ok=True)

    llm_client = build_openai_client(llm_base_url, llm_api_key)
    letta_client = Letta(base_url=normalize_url(letta_base_url))
    tracker = TokenTracker(output_file=token_file)

    with open(dataset_path, "r", encoding="utf-8") as file_obj:
        samples = json.load(file_obj)

    subset_samples = samples[start_idx:end_idx] if end_idx is not None else samples[start_idx:]
    results_by_id = collect_existing_results(output_path, subset_samples, cache_dir)
    aggregated_results = aggregate_results(output_path, subset_samples, cache_dir, results_by_id)

    print(f"MemGPT will process {len(subset_samples)} samples.")

    for index, sample in enumerate(subset_samples):
        absolute_index = start_idx + index
        sample_id = str(sample.get("sample_id", f"sample_{absolute_index}"))
        cache_file = build_cache_file(cache_dir, index, sample_id)

        if os.path.exists(cache_file):
            print(f"[{index + 1}/{len(subset_samples)}] Skip completed sample: {sample_id}")
            continue

        if sample_id in results_by_id:
            write_json(cache_file, results_by_id[sample_id])
            print(f"[{index + 1}/{len(subset_samples)}] Reused aggregated sample: {sample_id}")
            continue

        recall_file = os.path.join(cache_dir, f"{absolute_index:05d}_{safe_name(sample_id)}_recall.json")
        agent_id = None

        with tracker.stage(f"Sample {absolute_index}"):
            print(f"[{index + 1}/{len(subset_samples)}] Processing sample: {sample_id}")
            agent = letta_client.agents.create(
                name=f"memgpt_{safe_name(sample_id)}",
                llm_config=LlmConfig(
                    model=llm_model or DEFAULT_LLM_MODEL,
                    model_endpoint=normalize_url(llm_base_url),
                    model_endpoint_type="openai",
                    context_window=32768,
                ),
                embedding_config=EmbeddingConfig(
                    embedding_model=embedding_model_name or DEFAULT_EMBEDDING_MODEL_NAME,
                    embedding_endpoint=normalize_url(embedding_base_url),
                    embedding_endpoint_type=embedding_endpoint_type or DEFAULT_EMBEDDING_ENDPOINT_TYPE,
                    embedding_dim=embedding_dim or DEFAULT_EMBEDDING_DIM,
                ),
            )
            agent_id = agent.id

            try:
                load_conversation_into_archival_memory(agent_id, sample, letta_base_url)
                time.sleep(1.0)

                qa_results = []
                for qa_index, qa in enumerate(sample.get("qa", [])):
                    question = qa.get("question", "")
                    print(f"  [{qa_index + 1}/{len(sample.get('qa', []))}] {question}")
                    with tracker.stage(f"QA {qa_index}"):
                        response_text, retrieved = agent_answer(
                            agent_id=agent_id,
                            question=question,
                            llm_client=llm_client,
                            llm_model=llm_model or DEFAULT_LLM_MODEL,
                            letta_base_url=letta_base_url,
                            recall_file=recall_file,
                            retrieve_k=retrieve_k or DEFAULT_RETRIEVE_K,
                            tracker=tracker,
                            qa_idx=qa_index,
                        )

                    qa_results.append(
                        {
                            "question": question,
                            "answer": qa.get("answer", ""),
                            "category": qa.get("category"),
                            "response": response_text,
                            "retrieved": retrieved,
                        }
                    )

                sample_result = {"sample_id": sample_id, "qa": qa_results}
                write_json(cache_file, sample_result)
                results_by_id[sample_id] = sample_result
                aggregated_results = aggregate_results(output_path, subset_samples, cache_dir, results_by_id)
            finally:
                if agent_id is not None:
                    try:
                        letta_client.agents.delete(agent_id=agent_id)
                    except Exception as exc:
                        logger.warning("Failed to delete Letta agent %s: %s", agent_id, exc)
                if os.path.exists(recall_file):
                    os.remove(recall_file)

    tracker.save_to_json()
    print(f"MemGPT completed. Aggregated {len(aggregated_results)} samples.")
    return aggregated_results
