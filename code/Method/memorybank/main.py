import contextlib
import importlib
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
DEFAULT_LLM_API_KEY = "EMPTY"
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RETRIEVE_K = 10
DEFAULT_RATIO = 1.0
DEFAULT_TEMPERATURE = 0.3
DEFAULT_OUTPUT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Result/LOCOMO")
)

SUMMARY_SYSTEM_PROMPT = "You are an expert at summarizing conversations."
PERSONALITY_SYSTEM_PROMPT = "You are an expert in psychological analysis."
SUMMARY_USER_TEMPLATE = "Summarize the events and key information in the content:\n\n---\n{dialogue}\n---"
PERSONALITY_USER_TEMPLATE = (
    "Based on the following dialogue, please summarize the user's personality traits and emotions:\n\n---\n"
    "{dialogue}\n---"
)
QA_PROMPT_TEMPLATE = (
    "Based on the following context from previous conversations, answer the question"
    "(give concise and direct answer, do not give thinking process).\n\n"
    "Relevant Context:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n\n"
    "Answer: Answer with contents from the context whenever possible. Be concise and direct."
)


def resolve_path(path_value: Optional[str], base_dir: str) -> Optional[str]:
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def normalize_output_path(output_path: Optional[str]) -> str:
    if not output_path:
        return os.path.join(DEFAULT_OUTPUT_ROOT, "memorybank", "default", "result.json")
    if output_path.endswith(".json"):
        return output_path
    return os.path.join(output_path, "result.json")


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
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str, payload):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


class TokenTracker:
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.root = self._create_stage_node("root")
        self.stack = [self.root]
        self._patched_target = None
        self._patched_name = None
        self._original_call = None

    @staticmethod
    def _create_stage_node(name: str) -> Dict:
        return {
            "name": name,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0.0,
            "sub_stages": {},
        }

    @contextlib.contextmanager
    def stage(self, name: str):
        parent = self.stack[-1]
        if name not in parent["sub_stages"]:
            parent["sub_stages"][name] = self._create_stage_node(name)
        node = parent["sub_stages"][name]
        start_time = time.time()
        node["start_time"] = start_time
        self.stack.append(node)
        try:
            yield
        finally:
            end_time = time.time()
            node["end_time"] = end_time
            node["duration_seconds"] += end_time - start_time
            self.stack.pop()

    def _extract_usage(self, response) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is not None:
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
        if isinstance(response, dict):
            usage = response.get("usage", {}) or {}
            return {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def patch_openai(self, openai_module):
        if hasattr(openai_module, "ChatCompletion"):
            target = openai_module.ChatCompletion
            function_name = "create"
        else:
            resource_module = importlib.import_module("openai.resources.chat.completions")
            target = resource_module.Completions
            function_name = "create"

        original = getattr(target, function_name)
        tracker = self

        def wrapped(*args, **kwargs):
            response = original(*args, **kwargs)
            usage = tracker._extract_usage(response)
            current = tracker.stack[-1]
            current["prompt_tokens"] += usage["prompt_tokens"]
            current["completion_tokens"] += usage["completion_tokens"]
            current["total_tokens"] += usage["total_tokens"]
            return response

        self._patched_target = target
        self._patched_name = function_name
        self._original_call = original
        setattr(target, function_name, wrapped)

    def restore(self):
        if self._patched_target is not None and self._patched_name and self._original_call is not None:
            setattr(self._patched_target, self._patched_name, self._original_call)
        self._patched_target = None
        self._patched_name = None
        self._original_call = None

    def save_to_json(self):
        def aggregate(node: Dict):
            for child in node["sub_stages"].values():
                aggregate(child)
            node["prompt_tokens"] += sum(child["prompt_tokens"] for child in node["sub_stages"].values())
            node["completion_tokens"] += sum(child["completion_tokens"] for child in node["sub_stages"].values())
            node["total_tokens"] += sum(child["total_tokens"] for child in node["sub_stages"].values())

        aggregate(self.root)
        write_json(self.output_file, self.root)


def import_runtime_dependencies():
    try:
        import openai
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "MemoryBank requires `openai` and `sentence-transformers` to be installed in the current environment."
        ) from exc

    return {
        "openai": openai,
        "SentenceTransformer": SentenceTransformer,
    }


def call_chat_completion(openai_module, model_name: str, api_key: str, base_url: str, messages: List[Dict], temperature: float):
    if hasattr(openai_module, "ChatCompletion"):
        return openai_module.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=5000,
            api_key=api_key,
            api_base=base_url,
            timeout=60,
        )

    client_cls = getattr(openai_module, "OpenAI", None)
    if client_cls is None:
        raise RuntimeError("Unsupported OpenAI SDK version for MemoryBank runtime.")
    client = client_cls(api_key=api_key, base_url=base_url)
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=5000,
        timeout=60,
    )


def extract_response_text(response) -> str:
    if isinstance(response, dict):
        try:
            return str(response["choices"][0]["message"]["content"]).strip()
        except Exception:
            return str(response)

    try:
        return str(response.choices[0].message.content).strip()
    except Exception:
        return str(response).strip()


def format_turn_text(turn: Dict) -> str:
    text = str(turn.get("text", ""))
    caption = str(turn.get("blip_caption", "")).strip()
    if caption:
        return f"{text} (image description: {caption})"
    return text


def build_session_pairs(conversation_data: Dict, session_key: str, speaker_a: str, speaker_b: str) -> List[Dict]:
    session_dialogue = conversation_data.get(session_key) or []
    session_pairs: List[Dict] = []
    index = 0
    while index < len(session_dialogue) - 1:
        current_turn = session_dialogue[index]
        next_turn = session_dialogue[index + 1]
        is_pair = (
            current_turn.get("speaker") in {speaker_a, speaker_b}
            and next_turn.get("speaker") in {speaker_a, speaker_b}
            and current_turn.get("speaker") != next_turn.get("speaker")
        )
        if is_pair:
            session_pairs.append(
                {
                    "query": f"{current_turn.get('speaker')}: {format_turn_text(current_turn)}",
                    "response": f"{next_turn.get('speaker')}: {format_turn_text(next_turn)}",
                }
            )
            index += 2
        else:
            index += 1
    return session_pairs


def generate_chat_summary(openai_module, dialogues: List[Dict], model_name: str, api_key: str, base_url: str) -> str:
    dialogue_text = "\n".join(f"{item['query']}\n{item['response']}" for item in dialogues)
    response = call_chat_completion(
        openai_module,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": SUMMARY_USER_TEMPLATE.format(dialogue=dialogue_text)},
        ],
        temperature=DEFAULT_TEMPERATURE,
    )
    return extract_response_text(response)


def generate_personality_summary(openai_module, dialogues: List[Dict], model_name: str, api_key: str, base_url: str) -> str:
    dialogue_text = "\n".join(f"{item['query']}\n{item['response']}" for item in dialogues)
    response = call_chat_completion(
        openai_module,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        messages=[
            {"role": "system", "content": PERSONALITY_SYSTEM_PROMPT},
            {"role": "user", "content": PERSONALITY_USER_TEMPLATE.format(dialogue=dialogue_text)},
        ],
        temperature=DEFAULT_TEMPERATURE,
    )
    return extract_response_text(response)


def convert_sample_memory(
    openai_module,
    sample: Dict,
    model_name: str,
    api_key: str,
    base_url: str,
    tracker,
) -> Tuple[Dict, List[Dict]]:
    conversation_data = sample.get("conversation", {}) or {}
    speaker_a = conversation_data.get("speaker_a", "SpeakerA")
    speaker_b = conversation_data.get("speaker_b", "SpeakerB")

    history = {}
    session_summaries = {}
    session_personality = {}
    session_analyses = []

    session_keys = sorted(
        key for key in conversation_data.keys() if key.startswith("session_") and not key.endswith("_date_time")
    )

    for session_key in session_keys:
        session_time = conversation_data.get(f"{session_key}_date_time")
        if not session_time:
            continue
        session_pairs = build_session_pairs(conversation_data, session_key, speaker_a, speaker_b)
        if not session_pairs:
            continue

        history[session_time] = session_pairs
        with tracker.stage(f"{session_key}_summary"):
            summary = generate_chat_summary(openai_module, session_pairs, model_name, api_key, base_url)
        with tracker.stage(f"{session_key}_personality"):
            personality = generate_personality_summary(openai_module, session_pairs, model_name, api_key, base_url)

        session_summaries[session_time] = summary
        session_personality[session_time] = personality
        session_analyses.append(
            {
                "sample_id": sample.get("sample_id"),
                "session_id": session_time,
                "summary": summary,
                "personality": personality,
            }
        )

    return {
        "name": str(sample.get("sample_id", "sample")),
        "history": history,
        "summary": session_summaries,
        "personality": session_personality,
    }, session_analyses


def build_memory_docs(memory_data: Dict) -> List[str]:
    memory_docs: List[str] = []
    for session_time, conversations in (memory_data.get("history") or {}).items():
        for item in conversations:
            memory_docs.append(f"Session at {session_time}: {item.get('query', '')} -> {item.get('response', '')}")
    for session_time, summary_text in (memory_data.get("summary") or {}).items():
        memory_docs.append(f"Session {session_time} Summary: {summary_text}")
    return memory_docs


def retrieve_memory_texts(question: str, memory_docs: List[str], doc_embeddings, embedder, retrieve_k: int) -> List[str]:
    if not memory_docs:
        return []

    query_embedding = np.asarray(embedder.encode([question], show_progress_bar=False)[0], dtype=np.float32)
    doc_matrix = np.asarray(doc_embeddings, dtype=np.float32)
    doc_norms = np.linalg.norm(doc_matrix, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    denominator = np.maximum(doc_norms * max(query_norm, 1e-12), 1e-12)
    similarities = np.dot(doc_matrix, query_embedding) / denominator

    top_k = min(max(int(retrieve_k or DEFAULT_RETRIEVE_K), 1), len(memory_docs))
    top_indices = np.argsort(-similarities)[:top_k]
    return [memory_docs[index] for index in top_indices]


def answer_question(
    openai_module,
    question: str,
    retrieved_texts: List[str],
    model_name: str,
    api_key: str,
    base_url: str,
):
    context_text = "\n".join(retrieved_texts) if retrieved_texts else "No relevant memory found."
    response = call_chat_completion(
        openai_module,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        messages=[
            {
                "role": "user",
                "content": QA_PROMPT_TEMPLATE.format(context_str=context_text, query_str=question),
            }
        ],
        temperature=0.0,
    )
    return extract_response_text(response)


def load_dataset(dataset_path: str) -> List[Dict]:
    dataset = read_json(dataset_path, [])
    if not isinstance(dataset, list):
        raise ValueError(f"Expected dataset JSON array at {dataset_path}")
    return dataset


def apply_sample_slice(samples: List[Dict], start_idx: int, end_idx: Optional[int], ratio: float) -> List[Tuple[int, Dict]]:
    indexed_samples = list(enumerate(samples))
    sliced = indexed_samples[start_idx:end_idx] if end_idx is not None else indexed_samples[start_idx:]
    if ratio is None or ratio >= 1.0:
        return sliced
    if ratio <= 0:
        return []
    keep_count = max(1, int(len(sliced) * ratio)) if sliced else 0
    return sliced[:keep_count]


def build_cache_file(cache_dir: str, dataset_index: int, sample_id: str) -> str:
    return os.path.join(cache_dir, f"{dataset_index:05d}_{safe_name(sample_id)}.json")


def collect_existing_results(output_path: str, selected_samples: List[Tuple[int, Dict]], cache_dir: str) -> Dict[str, Dict]:
    existing = read_json(output_path, [])
    results_by_id = {}
    if isinstance(existing, list):
        for item in existing:
            sample_id = str(item.get("sample_id"))
            if sample_id:
                results_by_id[sample_id] = item

    for dataset_index, sample in selected_samples:
        sample_id = str(sample.get("sample_id", dataset_index))
        cache_file = build_cache_file(cache_dir, dataset_index, sample_id)
        if sample_id in results_by_id and not os.path.exists(cache_file):
            write_json(cache_file, results_by_id[sample_id])

    return results_by_id


def aggregate_results(output_path: str, selected_samples: List[Tuple[int, Dict]], cache_dir: str, results_by_id: Dict[str, Dict]) -> List[Dict]:
    aggregated = []
    for dataset_index, sample in selected_samples:
        sample_id = str(sample.get("sample_id", dataset_index))
        cache_file = build_cache_file(cache_dir, dataset_index, sample_id)
        if os.path.exists(cache_file):
            cached = read_json(cache_file, None)
            if cached:
                results_by_id[sample_id] = cached
        if sample_id in results_by_id:
            aggregated.append(results_by_id[sample_id])

    write_json(output_path, aggregated)
    return aggregated


def summarize_token_file(token_payload: Dict) -> Dict[str, int]:
    if not isinstance(token_payload, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": token_payload.get("prompt_tokens", 0) or 0,
        "completion_tokens": token_payload.get("completion_tokens", 0) or 0,
        "total_tokens": token_payload.get("total_tokens", 0) or 0,
    }


def aggregate_token_stats(token_file: str, sample_dirs: List[Tuple[str, str]]):
    aggregated = {
        "method": "memorybank",
        "sample_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "samples": [],
    }

    for sample_id, sample_token_file in sample_dirs:
        if not os.path.exists(sample_token_file):
            continue
        token_payload = read_json(sample_token_file, {})
        token_totals = summarize_token_file(token_payload)
        aggregated["sample_count"] += 1
        aggregated["prompt_tokens"] += token_totals["prompt_tokens"]
        aggregated["completion_tokens"] += token_totals["completion_tokens"]
        aggregated["total_tokens"] += token_totals["total_tokens"]
        aggregated["samples"].append(
            {
                "sample_id": sample_id,
                "token_file": sample_token_file,
                **token_totals,
            }
        )

    write_json(token_file, aggregated)


@contextlib.contextmanager
def tracker_stage(tracker, name: str):
    if tracker is None:
        yield
        return
    with tracker.stage(name):
        yield


def run_memorybank(
    dataset_path,
    output_path=None,
    token_file=None,
    llm_model=DEFAULT_LLM_MODEL,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
    retrieve_k=DEFAULT_RETRIEVE_K,
    ratio=DEFAULT_RATIO,
    start_idx=0,
    end_idx=None,
    config_path=None,
):
    base_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()
    dataset_path = resolve_path(dataset_path, base_dir)
    if not dataset_path:
        raise ValueError("memorybank requires 'dataset_path' in the config or CLI arguments.")

    output_path = normalize_output_path(resolve_path(output_path, base_dir))
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    token_file = resolve_path(token_file, base_dir) or os.path.join(output_dir, "token_tracker.json")
    cache_dir = os.path.join(output_dir, "_memorybank_cache")
    os.makedirs(cache_dir, exist_ok=True)

    runtime = import_runtime_dependencies()
    openai_module = runtime["openai"]
    os.environ["OPENAI_API_KEY"] = llm_api_key or DEFAULT_LLM_API_KEY
    os.environ["OPENAI_API_BASE"] = llm_base_url or DEFAULT_LLM_BASE_URL

    all_samples = load_dataset(dataset_path)
    selected_samples = apply_sample_slice(
        all_samples,
        start_idx=start_idx if start_idx is not None else 0,
        end_idx=end_idx,
        ratio=ratio if ratio is not None else DEFAULT_RATIO,
    )

    results_by_id = collect_existing_results(output_path, selected_samples, cache_dir)
    aggregated_results = aggregate_results(output_path, selected_samples, cache_dir, results_by_id)

    embedder = runtime["SentenceTransformer"](embedding_model_name or DEFAULT_EMBEDDING_MODEL_NAME)

    sample_token_files: List[Tuple[str, str]] = []

    for dataset_index, sample in selected_samples:
        sample_id = str(sample.get("sample_id", dataset_index))
        cache_file = build_cache_file(cache_dir, dataset_index, sample_id)
        sample_dir = os.path.join(output_dir, f"sample_{dataset_index}")
        os.makedirs(sample_dir, exist_ok=True)
        sample_token_file = os.path.join(sample_dir, "mem_l1_token_stats.json")
        sample_analysis_file = os.path.join(sample_dir, "all_sessions_analysis.json")
        sample_qa_file = os.path.join(sample_dir, "qa_results.json")
        sample_token_files.append((sample_id, sample_token_file))

        if os.path.exists(cache_file):
            print(f"[{dataset_index}] Skip completed sample: {sample_id}")
            continue

        if sample_id in results_by_id:
            write_json(cache_file, results_by_id[sample_id])
            if not os.path.exists(sample_qa_file):
                write_json(sample_qa_file, results_by_id[sample_id].get("qa", []))
            print(f"[{dataset_index}] Reused aggregated sample: {sample_id}")
            continue

        tracker = TokenTracker(output_file=sample_token_file)
        tracker.patch_openai(openai_module)

        try:
            with tracker_stage(tracker, f"sample_{dataset_index}"):
                memory_data, session_analyses = convert_sample_memory(
                    openai_module=openai_module,
                    sample=sample,
                    model_name=llm_model or DEFAULT_LLM_MODEL,
                    api_key=llm_api_key or DEFAULT_LLM_API_KEY,
                    base_url=llm_base_url or DEFAULT_LLM_BASE_URL,
                    tracker=tracker,
                )
                write_json(sample_analysis_file, session_analyses)

                memory_docs = build_memory_docs(memory_data)
                qa_results = []

                if memory_docs:
                    with tracker_stage(tracker, "build_index"):
                        doc_embeddings = np.asarray(
                            embedder.encode(memory_docs, show_progress_bar=False),
                            dtype=np.float32,
                        )
                else:
                    doc_embeddings = None

                for qa_index, qa in enumerate(sample.get("qa", [])):
                    question = qa.get("question", "")
                    gold_answer = qa.get("answer", "")
                    category = qa.get("category")

                    with tracker_stage(tracker, f"qa_{qa_index}"):
                        if doc_embeddings is None:
                            response = "No information available in memory."
                            retrieved = []
                        else:
                            try:
                                retrieved = retrieve_memory_texts(
                                    question=question,
                                    memory_docs=memory_docs,
                                    doc_embeddings=doc_embeddings,
                                    embedder=embedder,
                                    retrieve_k=retrieve_k if retrieve_k is not None else DEFAULT_RETRIEVE_K,
                                )
                                response = answer_question(
                                    openai_module=openai_module,
                                    question=question,
                                    retrieved_texts=retrieved,
                                    model_name=llm_model or DEFAULT_LLM_MODEL,
                                    api_key=llm_api_key or DEFAULT_LLM_API_KEY,
                                    base_url=llm_base_url or DEFAULT_LLM_BASE_URL,
                                )
                            except Exception as exc:
                                response = f"Error in generation: {exc}"
                                retrieved = []

                    qa_results.append(
                        {
                            "question": question,
                            "answer": gold_answer,
                            "category": category,
                            "response": response,
                            "retrieved": retrieved,
                        }
                    )
        finally:
            tracker.restore()
            tracker.save_to_json()

        sample_result = {"sample_id": sample_id, "qa": qa_results}
        write_json(sample_qa_file, qa_results)
        write_json(cache_file, sample_result)
        results_by_id[sample_id] = sample_result
        aggregated_results = aggregate_results(output_path, selected_samples, cache_dir, results_by_id)

    aggregate_token_stats(token_file, sample_token_files)
    return aggregated_results
