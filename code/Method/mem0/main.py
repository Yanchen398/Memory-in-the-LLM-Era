import json
import os
import socket
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from .token_tracker_mem0 import TokenTracker


DEFAULT_LLM_PROVIDER = "vllm"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
DEFAULT_LLM_API_KEY = "empty"
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_EMBEDDING_MODEL_NAME = "text-embedding-3-small"
DEFAULT_EMBEDDING_API_KEY = "empty"
DEFAULT_EMBEDDING_BASE_URL = "http://localhost:7999/v1"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_RETRIEVE_K = 10
DEFAULT_BATCH_SIZE = 2
DEFAULT_RATIO = 1.0
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "neo4j"


ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers.
2. Pay special attention to timestamps to determine the answer.
3. If the question asks about a specific event or fact, look for direct evidence in the memories.
4. If memories contain contradictory information, prioritize the most recent memory.
5. If the answer involves a relative time reference, convert it to a concrete date or year.
6. Focus only on the content of the memories from both speakers.
7. The answer should be less than 5-6 words. Do not include reasoning.

Memories for user {{speaker_1_user_id}}:

{{speaker_1_memories}}

Memories for user {{speaker_2_user_id}}:

{{speaker_2_memories}}

Question: {{question}}

Answer:
"""


ANSWER_PROMPT_GRAPH = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question. You also have access to graph relations extracted from each speaker's memories.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers.
2. Pay special attention to timestamps to determine the answer.
3. If the question asks about a specific event or fact, look for direct evidence in the memories.
4. If memories contain contradictory information, prioritize the most recent memory.
5. If the answer involves a relative time reference, convert it to a concrete date or year.
6. Use graph relations only as supporting context for entity relationships.
7. Focus only on the content of the memories from both speakers.
8. The answer should be less than 5-6 words. Do not include reasoning.

Memories for user {{speaker_1_user_id}}:

{{speaker_1_memories}}

Relations for user {{speaker_1_user_id}}:

{{speaker_1_graph_memories}}

Memories for user {{speaker_2_user_id}}:

{{speaker_2_memories}}

Relations for user {{speaker_2_user_id}}:

{{speaker_2_graph_memories}}

Question: {{question}}

Answer:
"""


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../Result/LOCOMO"))


def resolve_path(path_value: Optional[str], base_dir: str) -> Optional[str]:
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def normalize_output_path(output_path: Optional[str], method_name: str) -> str:
    if not output_path:
        return os.path.join(DEFAULT_OUTPUT_ROOT, method_name, "default", "result.json")
    if output_path.endswith(".json"):
        return output_path
    return os.path.join(output_path, "result.json")


def normalize_url(url: Optional[str]) -> str:
    return (url or "").rstrip("/")


def safe_name(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))
    return sanitized.strip("._") or "sample"


def import_mem0_runtime():
    try:
        from mem0 import Memory
        from mem0.vector_stores.faiss import FAISS
    except ImportError as exc:
        raise ImportError(
            "Mem0 requires the external `mem0` package in the current environment."
        ) from exc

    return Memory, FAISS


def set_embedding_runtime_env(
    llm_api_key: str,
    llm_base_url: str,
    embedding_api_key: Optional[str],
    embedding_base_url: Optional[str],
):
    effective_api_key = embedding_api_key or llm_api_key or DEFAULT_LLM_API_KEY
    effective_base_url = embedding_base_url or llm_base_url or DEFAULT_LLM_BASE_URL

    if effective_api_key:
        os.environ["OPENAI_API_KEY"] = effective_api_key
    if effective_base_url:
        os.environ["OPENAI_BASE_URL"] = effective_base_url


def parse_service_uri(uri: str) -> Tuple[str, int]:
    parsed = urlsplit(uri)
    host = parsed.hostname or "localhost"
    if parsed.port:
        return host, parsed.port
    if parsed.scheme in {"bolt", "neo4j"}:
        return host, 7687
    return host, 80


def is_service_reachable(uri: str, timeout: int = 3) -> bool:
    host, port = parse_service_uri(uri)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def build_llm_config(llm_provider: str, llm_model: str, llm_api_key: str, llm_base_url: str) -> Dict:
    llm_provider = llm_provider or DEFAULT_LLM_PROVIDER
    llm_base_url = normalize_url(llm_base_url or DEFAULT_LLM_BASE_URL)

    config = {
        "api_key": llm_api_key or DEFAULT_LLM_API_KEY,
        "model": llm_model or DEFAULT_LLM_MODEL,
    }

    if llm_provider == "vllm":
        config["vllm_base_url"] = llm_base_url
    elif llm_provider == "openai":
        config["openai_base_url"] = llm_base_url
    else:
        config["base_url"] = llm_base_url

    return {"provider": llm_provider, "config": config}


def build_memory_config(
    method_name: str,
    faiss_path: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
    embedding_dim: int,
    enable_graph: bool,
    neo4j_uri: Optional[str],
    neo4j_user: Optional[str],
    neo4j_password: Optional[str],
) -> Dict:
    config = {
        "vector_store": {
            "provider": "faiss",
            "config": {
                "path": faiss_path,
                "embedding_model_dims": embedding_dim,
            },
        },
        "llm": build_llm_config(llm_provider, llm_model, llm_api_key, llm_base_url),
    }

    if enable_graph:
        config["graph_store"] = {
            "provider": "neo4j",
            "config": {
                "url": neo4j_uri or DEFAULT_NEO4J_URI,
                "username": neo4j_user or DEFAULT_NEO4J_USER,
                "password": neo4j_password or DEFAULT_NEO4J_PASSWORD,
            },
        }

    return config


def load_dataset(dataset_path: str) -> List[Dict]:
    with open(dataset_path, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "samples" in data:
            return data["samples"]
        if "qa" in data and "conversation" in data:
            return [data]

    raise ValueError(f"Unsupported dataset format in {dataset_path}")


def apply_sample_slice(samples: List[Dict], start_idx: int, end_idx: Optional[int], ratio: float) -> List[Dict]:
    end_idx = len(samples) if end_idx is None or end_idx > len(samples) else end_idx
    subset = samples[start_idx:end_idx]
    if ratio is None:
        return subset
    if ratio <= 0.0 or ratio > 1.0:
        raise ValueError("ratio must be within (0.0, 1.0].")
    limit = max(1, int(len(subset) * ratio)) if subset else 0
    return subset[:limit]


def build_cache_file(cache_dir: str, index: int, sample_id: str) -> str:
    return os.path.join(cache_dir, f"{index:05d}_{safe_name(sample_id)}.json")


def build_token_file(token_dir: str, index: int, sample_id: str) -> str:
    return os.path.join(token_dir, f"{index:05d}_{safe_name(sample_id)}.json")


def load_json_if_exists(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(path: str, data):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


def collect_existing_results(output_path: str, subset_samples: List[Dict], cache_dir: str) -> Dict[str, Dict]:
    results_by_id: Dict[str, Dict] = {}

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


def aggregate_results(output_path: str, subset_samples: List[Dict], cache_dir: str, results_by_id: Dict[str, Dict]) -> List[Dict]:
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


def aggregate_token_stats(token_file: str, subset_samples: List[Dict], token_dir: str, method_name: str):
    sample_stats = {}
    total_prompt = 0
    total_completion = 0
    total_tokens = 0

    for index, sample in enumerate(subset_samples):
        sample_id = str(sample.get("sample_id", f"sample_{index}"))
        sample_token_file = build_token_file(token_dir, index, sample_id)
        if not os.path.exists(sample_token_file):
            continue

        stats = load_json_if_exists(sample_token_file, {})
        sample_stats[sample_id] = stats
        total_prompt += int(stats.get("prompt_tokens", 0) or 0)
        total_completion += int(stats.get("completion_tokens", 0) or 0)
        total_tokens += int(stats.get("total_tokens", 0) or 0)

    write_json(
        token_file,
        {
            "method": method_name,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "samples": sample_stats,
        },
    )


def format_memory_with_timestamp(memory_item: Dict) -> str:
    if not isinstance(memory_item, dict):
        return str(memory_item)

    memory_text = str(memory_item.get("memory", ""))
    metadata = memory_item.get("metadata") or {}
    timestamp = metadata.get("timestamp") or memory_item.get("created_at", "")
    speaker = metadata.get("speaker", "")

    if speaker and timestamp:
        return f"{speaker} at {timestamp}: {memory_text}"
    if timestamp:
        return f"[{timestamp}] {memory_text}"
    return memory_text


def format_relation(relation: Dict) -> str:
    if not isinstance(relation, dict):
        return str(relation)
    source = relation.get("source", "")
    relationship = relation.get("relationship", relation.get("relation", ""))
    destination = relation.get("destination", "")
    if source or relationship or destination:
        return f"{source} {relationship} {destination}".strip()
    return json.dumps(relation, ensure_ascii=False)


def extract_memory_results(search_result) -> List[Dict]:
    if isinstance(search_result, dict):
        results = search_result.get("results")
        if isinstance(results, list):
            return results
    if isinstance(search_result, list):
        return search_result
    return []


def extract_relation_results(search_result) -> List[Dict]:
    if isinstance(search_result, dict):
        relations = search_result.get("relations")
        if isinstance(relations, list):
            return relations
    return []


def build_prompt(
    question: str,
    speaker_a_user_id: str,
    speaker_b_user_id: str,
    speaker_a_memories: str,
    speaker_b_memories: str,
    speaker_a_graph: str,
    speaker_b_graph: str,
    enable_graph: bool,
) -> str:
    prompt = ANSWER_PROMPT_GRAPH if enable_graph else ANSWER_PROMPT
    prompt = prompt.replace("{{speaker_1_user_id}}", speaker_a_user_id)
    prompt = prompt.replace("{{speaker_1_memories}}", speaker_a_memories)
    prompt = prompt.replace("{{speaker_2_user_id}}", speaker_b_user_id)
    prompt = prompt.replace("{{speaker_2_memories}}", speaker_b_memories)
    prompt = prompt.replace("{{question}}", question)
    if enable_graph:
        prompt = prompt.replace("{{speaker_1_graph_memories}}", speaker_a_graph)
        prompt = prompt.replace("{{speaker_2_graph_memories}}", speaker_b_graph)
    return prompt


def maybe_set_token_tracker(memory, tracker: TokenTracker):
    llm = getattr(memory, "llm", None)
    if llm is not None and hasattr(llm, "set_token_tracker"):
        llm.set_token_tracker(tracker)


def create_memory_runtime(
    method_name: str,
    faiss_path: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
    embedding_dim: int,
    enable_graph: bool,
    neo4j_uri: Optional[str],
    neo4j_user: Optional[str],
    neo4j_password: Optional[str],
):
    Memory, FAISS = import_mem0_runtime()

    faiss_store = FAISS(
        collection_name=method_name,
        path=faiss_path,
        embedding_model_dims=embedding_dim,
    )
    if hasattr(faiss_store, "reset"):
        faiss_store.reset()

    memory = Memory.from_config(
        build_memory_config(
            method_name=method_name,
            faiss_path=faiss_path,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_dim=embedding_dim,
            enable_graph=enable_graph,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )
    )

    if hasattr(memory, "enable_graph"):
        memory.enable_graph = enable_graph
    if hasattr(memory, "reset"):
        memory.reset()

    return memory


def close_memory_runtime(memory):
    if memory is not None and hasattr(memory, "close"):
        memory.close()


def build_session_messages(conversation: Dict, speaker_a: str, speaker_b: str, key: str, timestamp: str) -> Tuple[List[Dict], List[Dict]]:
    speaker_a_messages: List[Dict] = []
    speaker_b_messages: List[Dict] = []

    for chat in conversation.get(key, []):
        chat_timestamp = chat.get("timestamp", timestamp)
        text = str(chat.get("text", ""))
        if chat.get("blip_caption"):
            text = f"{text} (image description: {chat['blip_caption']})"
        content = f"{chat['speaker']} at {chat_timestamp}: {text}"
        if chat["speaker"] == speaker_a:
            speaker_a_messages.append({"role": "user", "content": content})
            speaker_b_messages.append({"role": "assistant", "content": content})
        elif chat["speaker"] == speaker_b:
            speaker_a_messages.append({"role": "assistant", "content": content})
            speaker_b_messages.append({"role": "user", "content": content})
        else:
            raise ValueError(f"Unknown speaker: {chat['speaker']}")

    return speaker_a_messages, speaker_b_messages


def ingest_session_messages(memory, user_id: str, speaker: str, timestamp: str, key: str, messages: List[Dict], batch_size: int, tracker: Optional[TokenTracker]):
    for batch_index in range(0, len(messages), batch_size):
        batch = messages[batch_index:batch_index + batch_size]
        if not batch:
            continue

        stage_name = f"add_dialogue_{key}_{speaker}_{batch_index // max(batch_size, 1)}"
        if tracker is not None:
            with tracker.stage(stage_name):
                memory.add(
                    messages=batch,
                    user_id=user_id,
                    metadata={"timestamp": timestamp, "speaker": speaker},
                    infer=True,
                )
        else:
            memory.add(
                messages=batch,
                user_id=user_id,
                metadata={"timestamp": timestamp, "speaker": speaker},
                infer=True,
            )


def query_memories(memory, user_id: str, question: str, retrieve_k: int, stage_name: str, tracker: Optional[TokenTracker]):
    if tracker is not None:
        with tracker.stage(stage_name):
            return memory.search(query=question, user_id=user_id, limit=retrieve_k)
    return memory.search(query=question, user_id=user_id, limit=retrieve_k)


def generate_answer(memory, prompt: str, stage_name: str, tracker: Optional[TokenTracker]):
    if tracker is not None:
        with tracker.stage(stage_name):
            return memory.llm.generate_response(messages=[{"role": "user", "content": prompt}])
    return memory.llm.generate_response(messages=[{"role": "user", "content": prompt}])


def extract_response_text(response) -> str:
    if isinstance(response, dict) and "content" in response:
        return str(response["content"])
    return str(response)


def process_sample(
    memory,
    sample: Dict,
    batch_size: int,
    retrieve_k: int,
    enable_graph: bool,
    tracker: Optional[TokenTracker],
) -> Dict:
    conversation = sample.get("conversation", {})
    speaker_a = conversation.get("speaker_a", "SpeakerA")
    speaker_b = conversation.get("speaker_b", "SpeakerB")
    sample_id = str(sample.get("sample_id", "sample"))
    speaker_a_user_id = f"{speaker_a}_{sample_id}"
    speaker_b_user_id = f"{speaker_b}_{sample_id}"

    for key in sorted(conversation.keys()):
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue

        timestamp = conversation.get(f"{key}_date_time", "unknown time")
        messages_a, messages_b = build_session_messages(conversation, speaker_a, speaker_b, key, timestamp)
        ingest_session_messages(memory, speaker_a_user_id, speaker_a, timestamp, key, messages_a, batch_size, tracker)
        ingest_session_messages(memory, speaker_b_user_id, speaker_b, timestamp, key, messages_b, batch_size, tracker)

    qa_results = []
    for qa_index, qa in enumerate(sample.get("qa", [])):
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        category = qa.get("category")

        speaker_a_search = query_memories(
            memory,
            speaker_a_user_id,
            question,
            retrieve_k,
            stage_name=f"retrieval_speaker_a_q{qa_index}",
            tracker=tracker,
        )
        speaker_b_search = query_memories(
            memory,
            speaker_b_user_id,
            question,
            retrieve_k,
            stage_name=f"retrieval_speaker_b_q{qa_index}",
            tracker=tracker,
        )

        speaker_a_memories = [format_memory_with_timestamp(item) for item in extract_memory_results(speaker_a_search)]
        speaker_b_memories = [format_memory_with_timestamp(item) for item in extract_memory_results(speaker_b_search)]
        speaker_a_relations = [format_relation(item) for item in extract_relation_results(speaker_a_search)]
        speaker_b_relations = [format_relation(item) for item in extract_relation_results(speaker_b_search)]

        prompt = build_prompt(
            question=question,
            speaker_a_user_id=speaker_a_user_id,
            speaker_b_user_id=speaker_b_user_id,
            speaker_a_memories="\n".join(speaker_a_memories),
            speaker_b_memories="\n".join(speaker_b_memories),
            speaker_a_graph="\n".join(speaker_a_relations),
            speaker_b_graph="\n".join(speaker_b_relations),
            enable_graph=enable_graph,
        )
        response = generate_answer(
            memory,
            prompt,
            stage_name=f"qa_generation_q{qa_index}",
            tracker=tracker,
        )

        retrieved = list(speaker_a_memories) + list(speaker_b_memories)
        if enable_graph:
            retrieved.extend(speaker_a_relations)
            retrieved.extend(speaker_b_relations)

        qa_results.append(
            {
                "question": question,
                "answer": answer,
                "category": category,
                "response": extract_response_text(response),
                "retrieved": retrieved,
            }
        )

    return {"sample_id": sample_id, "qa": qa_results}


def run_mem0_variant(
    method_name: str,
    enable_graph: bool,
    dataset_path: str,
    output_path: Optional[str] = None,
    memory_path: Optional[str] = None,
    token_file: Optional[str] = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_api_key: str = DEFAULT_LLM_API_KEY,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    llm_provider: str = DEFAULT_LLM_PROVIDER,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_api_key: str = DEFAULT_EMBEDDING_API_KEY,
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    retrieve_k: int = DEFAULT_RETRIEVE_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    ratio: float = DEFAULT_RATIO,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    config_path: Optional[str] = None,
):
    base_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()

    dataset_path = resolve_path(dataset_path, base_dir)
    if not dataset_path:
        raise ValueError(f"{method_name} requires 'dataset_path' in the config or CLI arguments.")

    output_path = normalize_output_path(resolve_path(output_path, base_dir), method_name)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    memory_root = resolve_path(memory_path, base_dir) or os.path.join(output_dir, f"_{method_name}_runtime")
    faiss_path = os.path.join(memory_root, "faiss")
    cache_dir = os.path.join(output_dir, f"_{method_name}_cache")
    token_dir = os.path.join(output_dir, f"_{method_name}_token_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(token_dir, exist_ok=True)
    os.makedirs(memory_root, exist_ok=True)

    token_file = resolve_path(token_file, base_dir) or os.path.join(output_dir, "token_tracker.json")

    if enable_graph:
        effective_neo4j_uri = neo4j_uri or DEFAULT_NEO4J_URI
        if not is_service_reachable(effective_neo4j_uri):
            raise ConnectionError(
                f"Neo4j is not reachable at {effective_neo4j_uri}. Please start Neo4j before running {method_name}."
            )

    set_embedding_runtime_env(
        llm_api_key=llm_api_key or DEFAULT_LLM_API_KEY,
        llm_base_url=llm_base_url or DEFAULT_LLM_BASE_URL,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
    )

    # Keep these values available to external mem0 runtimes that read env vars.
    if embedding_model_name:
        os.environ["MEM0_EMBEDDING_MODEL_NAME"] = embedding_model_name
    os.environ["MEM0_EMBEDDING_DIM"] = str(embedding_dim or DEFAULT_EMBEDDING_DIM)

    all_samples = load_dataset(dataset_path)
    subset_samples = apply_sample_slice(
        all_samples,
        start_idx=start_idx if start_idx is not None else 0,
        end_idx=end_idx,
        ratio=ratio if ratio is not None else DEFAULT_RATIO,
    )

    results_by_id = collect_existing_results(output_path, subset_samples, cache_dir)
    aggregated_results = aggregate_results(output_path, subset_samples, cache_dir, results_by_id)

    for index, sample in enumerate(subset_samples):
        sample_id = str(sample.get("sample_id", f"sample_{index}"))
        cache_file = build_cache_file(cache_dir, index, sample_id)
        sample_token_file = build_token_file(token_dir, index, sample_id)

        if os.path.exists(cache_file):
            print(f"[{index + 1}/{len(subset_samples)}] Skip completed sample: {sample_id}")
            continue

        if sample_id in results_by_id:
            write_json(cache_file, results_by_id[sample_id])
            print(f"[{index + 1}/{len(subset_samples)}] Reused aggregated sample: {sample_id}")
            continue

        tracker = TokenTracker(output_file=sample_token_file)
        memory = None

        try:
            memory = create_memory_runtime(
                method_name=method_name,
                faiss_path=faiss_path,
                llm_provider=llm_provider or DEFAULT_LLM_PROVIDER,
                llm_model=llm_model or DEFAULT_LLM_MODEL,
                llm_api_key=llm_api_key or DEFAULT_LLM_API_KEY,
                llm_base_url=llm_base_url or DEFAULT_LLM_BASE_URL,
                embedding_dim=embedding_dim or DEFAULT_EMBEDDING_DIM,
                enable_graph=enable_graph,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
            )
            maybe_set_token_tracker(memory, tracker)
            sample_result = process_sample(
                memory=memory,
                sample=sample,
                batch_size=batch_size if batch_size is not None else DEFAULT_BATCH_SIZE,
                retrieve_k=retrieve_k if retrieve_k is not None else DEFAULT_RETRIEVE_K,
                enable_graph=enable_graph,
                tracker=tracker,
            )
            write_json(cache_file, sample_result)
            results_by_id[sample_id] = sample_result
            aggregated_results = aggregate_results(output_path, subset_samples, cache_dir, results_by_id)
            tracker.save()
        finally:
            close_memory_runtime(memory)

    aggregate_token_stats(token_file, subset_samples, token_dir, method_name)
    return aggregated_results


def run_mem0(
    dataset_path,
    output_path=None,
    memory_path=None,
    token_file=None,
    llm_model=DEFAULT_LLM_MODEL,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    llm_provider=DEFAULT_LLM_PROVIDER,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_api_key=DEFAULT_EMBEDDING_API_KEY,
    embedding_base_url=DEFAULT_EMBEDDING_BASE_URL,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    retrieve_k=DEFAULT_RETRIEVE_K,
    batch_size=DEFAULT_BATCH_SIZE,
    ratio=DEFAULT_RATIO,
    start_idx=0,
    end_idx=None,
    config_path=None,
):
    return run_mem0_variant(
        method_name="mem0",
        enable_graph=False,
        dataset_path=dataset_path,
        output_path=output_path,
        memory_path=memory_path,
        token_file=token_file,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_provider=llm_provider,
        embedding_model_name=embedding_model_name,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        embedding_dim=embedding_dim,
        retrieve_k=retrieve_k,
        batch_size=batch_size,
        ratio=ratio,
        start_idx=start_idx,
        end_idx=end_idx,
        config_path=config_path,
    )


def run_mem0g(
    dataset_path,
    output_path=None,
    memory_path=None,
    token_file=None,
    llm_model=DEFAULT_LLM_MODEL,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    llm_provider=DEFAULT_LLM_PROVIDER,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_api_key=DEFAULT_EMBEDDING_API_KEY,
    embedding_base_url=DEFAULT_EMBEDDING_BASE_URL,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    retrieve_k=DEFAULT_RETRIEVE_K,
    batch_size=DEFAULT_BATCH_SIZE,
    ratio=DEFAULT_RATIO,
    start_idx=0,
    end_idx=None,
    neo4j_uri=DEFAULT_NEO4J_URI,
    neo4j_user=DEFAULT_NEO4J_USER,
    neo4j_password=DEFAULT_NEO4J_PASSWORD,
    config_path=None,
):
    return run_mem0_variant(
        method_name="mem0g",
        enable_graph=True,
        dataset_path=dataset_path,
        output_path=output_path,
        memory_path=memory_path,
        token_file=token_file,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_provider=llm_provider,
        embedding_model_name=embedding_model_name,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        embedding_dim=embedding_dim,
        retrieve_k=retrieve_k,
        batch_size=batch_size,
        ratio=ratio,
        start_idx=start_idx,
        end_idx=end_idx,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        config_path=config_path,
    )
