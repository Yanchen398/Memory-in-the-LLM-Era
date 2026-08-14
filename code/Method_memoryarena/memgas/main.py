import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from .config import MemoryConfig
from .token_tracker import TokenTracker
from ..dataset_hygiene import format_turn_text, natural_session_keys, raw_result_path


DEFAULT_LLM_MODEL = "Qwen3.5-9B"
DEFAULT_LLM_API_KEY = "empty"
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_LLM_PROVIDER = "vllm"
DEFAULT_EMBEDDER = "minilm"
DEFAULT_RETRIEVE_K = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_RATIO = 1.0
DEFAULT_MODE = "memgas"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
DEFAULT_OUTPUT_ROOT = os.path.abspath(os.path.join(CODE_DIR, "Result/LOCOMO"))


PROMPT_G = """
Answer the Question using only the Relevant Memory Evidence below.
The evidence contains memories about the user and is authoritative for this task.
First-person words such as "I", "me", and "my" in the Question refer to that user.
If the evidence directly provides the answer, state it directly even when it is personal information.
Never claim that you lack access to personal records when the evidence contains the answer.
Answer concisely without explaining these instructions or mentioning the evidence.

Relevant Memory Evidence:
{retrieved_texts}

Question Date: {question_date}
Question: {question}
Answer:
""".strip()

PROMPT_MULTIGRAN = """
Extract evidence from the History Dialogs that directly helps answer the Question.
Treat the dialogs as memories about the user; first-person words in the Question refer to that user.
Return only the relevant original facts, values, dates, names, or short dialog spans.
Preserve original tokens and do not answer with general advice or claims about lacking access.
Remove irrelevant turns, redundant information, and assistant boilerplate.
If there is no relevant evidence, return exactly: INSUFFICIENT_EVIDENCE

History Dialogs: {retrieved_texts}

Question Date: {question_date}
Question: {question}
Filtered Evidence:
""".strip()

INSUFFICIENT_EVIDENCE_MARKERS = (
    "insufficient_evidence",
    "no relevant evidence",
    "provided history dialogs do not contain",
    "cannot answer this question based on the provided",
)

ACCESS_REFUSAL_MARKERS = (
    "don't have access",
    "do not have access",
    "cannot access",
    "can't access",
    "unable to access",
    "no access to",
)


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


def normalize_output_path(output_path: Optional[str]) -> str:
    if not output_path:
        return os.path.join(DEFAULT_OUTPUT_ROOT, "memgas", "default", "result.json")
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
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(path: str, payload):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def load_dataset(dataset_path: str) -> List[Dict]:
    data = read_json(dataset_path, [])
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "samples" in data and isinstance(data["samples"], list):
            return data["samples"]
        if "qa" in data and "conversation" in data:
            return [data]
    raise ValueError(f"Unsupported LOCOMO dataset format in {dataset_path}")


def apply_sample_slice(samples: List[Dict], start_idx: int, end_idx: Optional[int], ratio: float) -> List[Tuple[int, Dict]]:
    indexed_samples = list(enumerate(samples))
    sliced = indexed_samples[start_idx:end_idx] if end_idx is not None else indexed_samples[start_idx:]
    if ratio is None or ratio >= 1.0:
        return sliced
    if ratio <= 0:
        return []
    keep_count = max(1, int(len(sliced) * ratio)) if sliced else 0
    return sliced[:keep_count]


def parse_retrieve_top_ks(value) -> List[int]:
    if value is None:
        return [DEFAULT_RETRIEVE_K]
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    top_ks: List[int] = []
    for part in parts:
        k = int(part)
        if k <= 0:
            raise ValueError(f"retrieve top-k values must be positive, got {k}")
        if k not in top_ks:
            top_ks.append(k)
    return top_ks or [DEFAULT_RETRIEVE_K]


def top_k_output_path(base_output_path: str, top_k: int) -> str:
    output_dir = os.path.dirname(base_output_path)
    output_name = os.path.basename(raw_result_path(base_output_path))
    return os.path.join(output_dir, f"top_k_{top_k}", output_name)


def top_k_cache_dir(base_output_path: str, top_k: int) -> str:
    return os.path.join(os.path.dirname(top_k_output_path(base_output_path, top_k)), "_memgas_cache")


def collect_stage_tokens(node: Dict, stage_name: str) -> Dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not isinstance(node, dict):
        return totals
    if node.get("name") == stage_name:
        return summarize_token_file(node)
    for child in (node.get("sub_stages") or {}).values():
        child_totals = collect_stage_tokens(child, stage_name)
        for key in totals:
            totals[key] += child_totals[key]
    return totals


def build_cache_file(cache_dir: str, dataset_index: int, sample_id: str) -> str:
    return os.path.join(cache_dir, f"{dataset_index:05d}_{safe_name(sample_id)}.json")


def build_token_file(token_dir: str, dataset_index: int, sample_id: str) -> str:
    return os.path.join(token_dir, f"{dataset_index:05d}_{safe_name(sample_id)}.json")


def get_sample_identifier(sample: Dict, dataset_index: int) -> str:
    return str(sample.get("sample_id") or sample.get("conversation_id") or dataset_index)


def collect_existing_results(output_path: str, selected_samples: List[Tuple[int, Dict]], cache_dir: str) -> Dict[str, Dict]:
    results_by_id: Dict[str, Dict] = {}
    existing_output = read_json(output_path, [])
    if isinstance(existing_output, list):
        for item in existing_output:
            sample_id = item.get("sample_id")
            if sample_id is not None:
                results_by_id[str(sample_id)] = item

    for dataset_index, sample in selected_samples:
        sample_id = get_sample_identifier(sample, dataset_index)
        cache_file = build_cache_file(cache_dir, dataset_index, sample_id)
        if os.path.exists(cache_file):
            results_by_id[sample_id] = read_json(cache_file, {})
    return results_by_id


def aggregate_results(
    output_path: str,
    selected_samples: List[Tuple[int, Dict]],
    cache_dir: str,
    results_by_id: Dict[str, Dict],
) -> List[Dict]:
    aggregated = []
    for dataset_index, sample in selected_samples:
        sample_id = get_sample_identifier(sample, dataset_index)
        cache_file = build_cache_file(cache_dir, dataset_index, sample_id)
        if os.path.exists(cache_file):
            results_by_id[sample_id] = read_json(cache_file, {})
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


def aggregate_token_stats(
    token_file: str,
    selected_samples: List[Tuple[int, Dict]],
    token_dir: str,
    retrieve_top_ks: Optional[List[int]] = None,
):
    aggregated = {
        "method": "memgas",
        "sample_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "samples": [],
        "top_k": {str(k): {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0} for k in (retrieve_top_ks or [])},
    }
    for dataset_index, sample in selected_samples:
        sample_id = get_sample_identifier(sample, dataset_index)
        sample_token_file = build_token_file(token_dir, dataset_index, sample_id)
        if not os.path.exists(sample_token_file):
            continue
        token_payload = read_json(sample_token_file, {})
        stats = summarize_token_file(token_payload)
        aggregated["sample_count"] += 1
        aggregated["prompt_tokens"] += stats["prompt_tokens"]
        aggregated["completion_tokens"] += stats["completion_tokens"]
        aggregated["total_tokens"] += stats["total_tokens"]
        sample_entry = {"sample_id": sample_id, "token_file": sample_token_file, **stats, "top_k": {}}
        for k in retrieve_top_ks or []:
            stage_stats = collect_stage_tokens(token_payload, f"top_k_{k}")
            sample_entry["top_k"][str(k)] = stage_stats
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                aggregated["top_k"][str(k)][key] += stage_stats[key]
        aggregated["samples"].append(sample_entry)
    write_json(token_file, aggregated)


def merge_turn_pairs(turns: List[str]) -> List[str]:
    merged = []
    for index in range(0, len(turns), 2):
        if index + 1 < len(turns):
            merged.append(turns[index] + "\n" + turns[index + 1])
        else:
            merged.append(turns[index])
    return merged


def convert_original_locomo_sample(sample: Dict) -> Dict:
    qa_items = []
    for qa in sample.get("qa", []):
        answer = qa.get("adversarial_answer", qa.get("answer", ""))
        answer_session_ids = []
        for evidence in qa.get("evidence", []) or []:
            try:
                int(str(evidence).split(":")[1])
            except Exception:
                continue
            answer_session_ids.append(str(evidence).replace("D", "session_").split(":")[0])
        qa_items.append(
            {
                "question": qa.get("question", ""),
                "question_type": qa.get("category"),
                "question_date": None,
                "answer": answer,
                "answer_session_ids": answer_session_ids,
            }
        )

    conversation = sample.get("conversation", {}) or {}
    session_ids = []
    session_dates = []
    sessions = []
    for session_id in natural_session_keys(conversation):
        turns = []
        for dialog in conversation.get(session_id, []) or []:
            speaker = dialog.get("speaker", "Speaker")
            turns.append(f"[{speaker}]: {format_turn_text(dialog)}")
        merged_turns = merge_turn_pairs(turns)
        if not merged_turns:
            continue
        session_ids.append(session_id)
        session_dates.append(conversation.get(f"{session_id}_date_time"))
        sessions.append(merged_turns)

    return {
        "conversation_id": sample.get("sample_id"),
        "qa": qa_items,
        "sessions_ids": session_ids,
        "sessions_dates": session_dates,
        "sessions": sessions,
    }


def normalize_memgas_sample(sample: Dict) -> Dict:
    if {"conversation_id", "sessions_ids", "sessions_dates", "sessions"}.issubset(sample.keys()):
        return sample
    if "conversation" in sample:
        return convert_original_locomo_sample(sample)
    raise ValueError("Unsupported MemGAS sample format: expected processed MemGAS data or original LOCOMO data.")


def format_hit(hit: Dict) -> str:
    metadata = hit.get("metadata") or {}
    timestamp = metadata.get("timestamp", "unknown time")
    session_text = str(hit.get("session") or [])
    summary = hit.get("summary", "")
    keywords = "; ".join(hit.get("keywords") or [])
    return (
        f"\n### Session Date: {timestamp}\n"
        f"Session Content:\n{session_text}\n\n"
        f"Session Summary:\n{summary}\n"
        f"Session Keyword:\n{keywords}\n"
    )


def complete_with_tracker(mem: Any, prompt: str, tracker: Optional[TokenTracker], stage_name: str) -> str:
    if tracker is None:
        return mem.llm._complete(prompt)
    with tracker.stage(stage_name):
        return mem.llm._complete(prompt)

def evidence_is_insufficient(text: str) -> bool:
    normalized = (text or "").strip().lower()
    return not normalized or any(marker in normalized for marker in INSUFFICIENT_EVIDENCE_MARKERS + ACCESS_REFUSAL_MARKERS)


def response_is_access_refusal(text: str) -> bool:
    normalized = (text or "").strip().lower()
    return any(marker in normalized for marker in ACCESS_REFUSAL_MARKERS)


def filter_retrieved_context(
    mem: Any,
    question: str,
    question_date: Any,
    retrieved_context: str,
    tracker: Optional[TokenTracker],
) -> str:
    if not retrieved_context.strip():
        return ""
    prompt = PROMPT_MULTIGRAN.format(
        retrieved_texts=retrieved_context,
        question_date=question_date,
        question=question,
    )
    filtered = complete_with_tracker(mem, prompt, tracker, "context_filter")
    if evidence_is_insufficient(filtered):
        return "INSUFFICIENT_EVIDENCE"
    return filtered.strip()


def generate_answer(
    mem: Any,
    question: str,
    question_date: Any,
    filtered_context: str,
    hits: List[Dict],
    tracker: Optional[TokenTracker],
) -> str:
    context = (filtered_context or "").strip()
    if evidence_is_insufficient(context):
        return "Insufficient context to answer."
    prompt = PROMPT_G.format(
        retrieved_texts=context,
        question_date=question_date,
        question=question,
    )
    response = complete_with_tracker(mem, prompt, tracker, "answer_generation")
    if response_is_access_refusal(response):
        return context
    return response.strip()


def process_sample(
    sample: Dict,
    sample_storage_dir: str,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
    llm_provider: str,
    embedder: str,
    device: Optional[str],
    batch_size: int,
    embedder_api_key: str,
    embedder_base_url: Optional[str],
    embedder_model: Optional[str],
    embedder_max_tokens: int,
    llm_max_tokens: int,
    llm_temperature: float,
    llm_max_retries: int,
    llm_retry_wait_sec: float,
    llm_context_window: int,
    llm_prompt_token_buffer: int,
    llm_use_qwen_thinking_control: bool,
    retrieve_top_ks: List[int],
    mode: str,
    mem_threshold: int,
    n_components: int,
    num_seednodes: int,
    damping: float,
    router_temp: float,
    tracker: Optional[TokenTracker],
    record_latency: bool,
) -> Dict:
    try:
        from .memory import MemGASMemory
    except ImportError as exc:
        raise ImportError(
            "MemGAS requires `torch`, `sentence-transformers`, `transformers`, "
            "`python-igraph`, `scikit-learn`, and `openai` in the current environment."
        ) from exc

    if os.path.exists(sample_storage_dir):
        shutil.rmtree(sample_storage_dir)

    config = MemoryConfig(
        storage_dir=sample_storage_dir,
        embedder=embedder,
        device=device,
        batch_size=batch_size,
        embedder_api_key=embedder_api_key,
        embedder_base_url=embedder_base_url,
        embedder_model=embedder_model,
        embedder_max_tokens=embedder_max_tokens,
        llm_model=llm_model,
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_max_tokens=llm_max_tokens,
        llm_temperature=llm_temperature,
        llm_max_retries=llm_max_retries,
        llm_retry_wait_sec=llm_retry_wait_sec,
        llm_context_window=llm_context_window,
        llm_prompt_token_buffer=llm_prompt_token_buffer,
        llm_use_qwen_thinking_control=llm_use_qwen_thinking_control,
        default_mode=mode,
        mem_threshold=mem_threshold,
        n_components=n_components,
        num_seednodes=num_seednodes,
        damping=damping,
        router_temp=router_temp,
        auto_save=True,
    )
    mem = MemGASMemory(config)

    processed_sample = normalize_memgas_sample(sample)
    sample_id = str(processed_sample.get("conversation_id", sample.get("sample_id", "sample")))
    session_ids = processed_sample.get("sessions_ids", []) or []
    session_dates = processed_sample.get("sessions_dates", []) or []
    sessions = processed_sample.get("sessions", []) or []

    for session_id, timestamp, session_texts in zip(session_ids, session_dates, sessions):
        stage_name = f"add_{session_id}"
        if tracker is None:
            mem.add(
                session=session_texts,
                conversation_id=sample_id,
                metadata={"timestamp": timestamp, "session_key": session_id},
            )
        else:
            with tracker.stage(stage_name):
                mem.add(
                    session=session_texts,
                    conversation_id=sample_id,
                    metadata={"timestamp": timestamp, "session_key": session_id},
                )

    results_by_top_k = {top_k: [] for top_k in retrieve_top_ks}
    for top_k in retrieve_top_ks:
        for qa_index, qa in enumerate(processed_sample.get("qa", [])):
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            question_type = qa.get("question_type", qa.get("category"))
            question_date = qa.get("question_date")

            retrieval_start = time.perf_counter()
            if tracker is None:
                hits = mem.retrieve(question, topk=top_k, conversation_id=sample_id, mode=mode)
            else:
                with tracker.stage(f"top_k_{top_k}"):
                    with tracker.stage(f"retrieval_q{qa_index}"):
                        hits = mem.retrieve(question, topk=top_k, conversation_id=sample_id, mode=mode)
            retrieval_latency = time.perf_counter() - retrieval_start

            retrieved = [format_hit(hit) for hit in hits]
            stage_tracker = tracker
            if tracker is not None:
                top_k_stage = tracker.stage(f"top_k_{top_k}")
                top_k_stage.__enter__()
            else:
                top_k_stage = None
            try:
                filtered_context = filter_retrieved_context(
                    mem=mem,
                    question=question,
                    question_date=question_date,
                    retrieved_context="".join(retrieved),
                    tracker=stage_tracker,
                )
                response = generate_answer(
                    mem=mem,
                    question=question,
                    question_date=question_date,
                    filtered_context=filtered_context,
                    hits=hits,
                    tracker=stage_tracker,
                )
            finally:
                if top_k_stage is not None:
                    top_k_stage.__exit__(None, None, None)

            qa_result = {
                "question": question,
                "answer": answer,
                "category": question_type,
                "question_type": question_type,
                "question_date": question_date,
                "answer_session_ids": qa.get("answer_session_ids"),
                "response": response,
                "retrieved": retrieved,
                "filtered_retrieved": filtered_context,
                "retrieve_top_k": top_k,
            }
            if record_latency:
                qa_result.update(
                    {
                        "retrieval_latency_seconds": retrieval_latency,
                        "retrieval_latency_ms": retrieval_latency * 1000.0,
                    }
                )
            results_by_top_k[top_k].append(qa_result)

    return {
        "sample_id": sample_id,
        "qa_by_top_k": results_by_top_k,
    }


def _process_sample_job(job: Dict) -> Dict:
    tracker = None
    if job["track_tokens"]:
        tracker = TokenTracker(output_file=job["sample_token_file"])
        tracker.patch_openai()

    try:
        if tracker is not None:
            with tracker.stage(f"sample_{job['dataset_index']}"):
                sample_result = process_sample(tracker=tracker, **job["process_kwargs"])
        else:
            sample_result = process_sample(tracker=None, **job["process_kwargs"])
    finally:
        if tracker is not None:
            tracker.restore()
            tracker.save_to_json()

    for top_k, cache_file in job["cache_files"].items():
        top_k_sample_result = {
            "sample_id": sample_result["sample_id"],
            "qa": sample_result["qa_by_top_k"].get(top_k, []),
        }
        write_json(cache_file, top_k_sample_result)

    return {
        "dataset_index": job["dataset_index"],
        "sample_id": job["sample_id"],
        "ordinal": job["ordinal"],
    }


def run_memgas(
    dataset_path,
    output_path=None,
    memory_path=None,
    token_file=None,
    llm_model=DEFAULT_LLM_MODEL,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    llm_provider=DEFAULT_LLM_PROVIDER,
    embedder=DEFAULT_EMBEDDER,
    device=None,
    batch_size=DEFAULT_BATCH_SIZE,
    embedder_api_key="EMPTY",
    embedder_base_url=None,
    embedder_model=None,
    embedder_max_tokens=256,
    llm_max_tokens=500,
    llm_temperature=0.0,
    llm_max_retries=3,
    llm_retry_wait_sec=2.0,
    llm_context_window=20000,
    llm_prompt_token_buffer=128,
    llm_use_qwen_thinking_control=True,
    retrieve_k=DEFAULT_RETRIEVE_K,
    retrieve_top_ks=None,
    auto_eval=False,
    eval_dataset="loco",
    eval_embedding_model=None,
    eval_result_path=None,
    ratio=DEFAULT_RATIO,
    start_idx=0,
    end_idx=None,
    mode=DEFAULT_MODE,
    mem_threshold=30,
    n_components=2,
    num_seednodes=15,
    damping=0.1,
    router_temp=0.2,
    num_workers=1,
    track_tokens=True,
    record_latency=True,
    config_path=None,
):
    base_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()

    dataset_path = resolve_path(dataset_path, base_dir)
    if not dataset_path:
        raise ValueError("MemGAS requires 'dataset_path' in the config or CLI arguments.")

    output_path = normalize_output_path(resolve_path(output_path, base_dir))
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    retrieve_top_ks = parse_retrieve_top_ks(retrieve_top_ks if retrieve_top_ks is not None else retrieve_k)
    top_k_paths = {top_k: top_k_output_path(output_path, top_k) for top_k in retrieve_top_ks}
    top_k_cache_dirs = {top_k: top_k_cache_dir(output_path, top_k) for top_k in retrieve_top_ks}

    memory_root = resolve_path(memory_path, base_dir) or os.path.join(output_dir, "_memgas_runtime")
    token_dir = os.path.join(output_dir, "_memgas_token_cache")
    os.makedirs(memory_root, exist_ok=True)
    os.makedirs(token_dir, exist_ok=True)
    for top_k_path in top_k_paths.values():
        os.makedirs(os.path.dirname(top_k_path), exist_ok=True)
    for cache_dir in top_k_cache_dirs.values():
        os.makedirs(cache_dir, exist_ok=True)

    token_file = resolve_path(token_file, base_dir) or os.path.join(output_dir, "token_tracker.json")

    all_samples = load_dataset(dataset_path)
    selected_samples = apply_sample_slice(
        all_samples,
        start_idx=start_idx if start_idx is not None else 0,
        end_idx=end_idx,
        ratio=ratio if ratio is not None else DEFAULT_RATIO,
    )

    results_by_top_k = {
        top_k: collect_existing_results(top_k_paths[top_k], selected_samples, top_k_cache_dirs[top_k])
        for top_k in retrieve_top_ks
    }
    aggregated_by_top_k = {
        top_k: aggregate_results(top_k_paths[top_k], selected_samples, top_k_cache_dirs[top_k], results_by_top_k[top_k])
        for top_k in retrieve_top_ks
    }

    process_defaults = {
        "llm_model": llm_model or DEFAULT_LLM_MODEL,
        "llm_api_key": llm_api_key or DEFAULT_LLM_API_KEY,
        "llm_base_url": llm_base_url or DEFAULT_LLM_BASE_URL,
        "llm_provider": llm_provider or DEFAULT_LLM_PROVIDER,
        "embedder": embedder or DEFAULT_EMBEDDER,
        "device": device,
        "batch_size": batch_size if batch_size is not None else DEFAULT_BATCH_SIZE,
        "embedder_api_key": embedder_api_key or "EMPTY",
        "embedder_base_url": embedder_base_url,
        "embedder_model": embedder_model,
        "embedder_max_tokens": embedder_max_tokens if embedder_max_tokens is not None else 256,
        "llm_max_tokens": llm_max_tokens if llm_max_tokens is not None else 500,
        "llm_temperature": llm_temperature if llm_temperature is not None else 0.0,
        "llm_max_retries": llm_max_retries if llm_max_retries is not None else 3,
        "llm_retry_wait_sec": llm_retry_wait_sec if llm_retry_wait_sec is not None else 2.0,
        "llm_context_window": llm_context_window if llm_context_window is not None else 20000,
        "llm_prompt_token_buffer": llm_prompt_token_buffer if llm_prompt_token_buffer is not None else 128,
        "llm_use_qwen_thinking_control": bool(llm_use_qwen_thinking_control),
        "retrieve_top_ks": retrieve_top_ks,
        "mode": mode or DEFAULT_MODE,
        "mem_threshold": mem_threshold if mem_threshold is not None else 30,
        "n_components": n_components if n_components is not None else 2,
        "num_seednodes": num_seednodes if num_seednodes is not None else 15,
        "damping": damping if damping is not None else 0.1,
        "router_temp": router_temp if router_temp is not None else 0.2,
        "record_latency": bool(record_latency),
    }

    if int(num_workers or 1) > 1:
        jobs = []
        for ordinal, (dataset_index, sample) in enumerate(selected_samples, start=1):
            sample_id = get_sample_identifier(sample, dataset_index)
            sample_token_file = build_token_file(token_dir, dataset_index, sample_id)
            sample_storage_dir = os.path.join(memory_root, f"{dataset_index:05d}_{safe_name(sample_id)}")
            cache_files = {
                top_k: build_cache_file(top_k_cache_dirs[top_k], dataset_index, sample_id)
                for top_k in retrieve_top_ks
            }
            if all(os.path.exists(cache_file) for cache_file in cache_files.values()):
                continue
            process_kwargs = dict(process_defaults)
            process_kwargs.update(
                sample=sample,
                sample_storage_dir=sample_storage_dir,
            )
            jobs.append(
                {
                    "ordinal": ordinal,
                    "dataset_index": dataset_index,
                    "sample_id": sample_id,
                    "sample_token_file": sample_token_file,
                    "cache_files": cache_files,
                    "track_tokens": bool(track_tokens),
                    "process_kwargs": process_kwargs,
                }
            )

        workers = min(int(num_workers), len(jobs)) if jobs else 0
        print(f"Starting sample-level multiprocessing with {workers} workers for {len(jobs)} samples.")
        if jobs:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
            ) as executor:
                futures = [
                    executor.submit(_process_sample_job, job)
                    for job in jobs
                ]
                completed = 0
                for future in as_completed(futures):
                    result = future.result()
                    completed += 1
                    print(f"[parallel {completed}/{len(jobs)}] Completed sample: {result['sample_id']}")

        for top_k in retrieve_top_ks:
            aggregated_by_top_k[top_k] = aggregate_results(
                top_k_paths[top_k],
                selected_samples,
                top_k_cache_dirs[top_k],
                results_by_top_k[top_k],
            )

    for ordinal, (dataset_index, sample) in enumerate(selected_samples, start=1):
        sample_id = get_sample_identifier(sample, dataset_index)
        sample_token_file = build_token_file(token_dir, dataset_index, sample_id)
        sample_storage_dir = os.path.join(memory_root, f"{dataset_index:05d}_{safe_name(sample_id)}")
        cache_files = {
            top_k: build_cache_file(top_k_cache_dirs[top_k], dataset_index, sample_id)
            for top_k in retrieve_top_ks
        }

        if all(os.path.exists(cache_file) for cache_file in cache_files.values()):
            print(f"[{ordinal}/{len(selected_samples)}] Skip completed sample for all top-k values: {sample_id}")
            continue

        tracker = TokenTracker(output_file=sample_token_file)
        tracker.patch_openai()
        try:
            with tracker.stage(f"sample_{dataset_index}"):
                sample_result = process_sample(
                    sample=sample,
                    sample_storage_dir=sample_storage_dir,
                    llm_model=llm_model or DEFAULT_LLM_MODEL,
                    llm_api_key=llm_api_key or DEFAULT_LLM_API_KEY,
                    llm_base_url=llm_base_url or DEFAULT_LLM_BASE_URL,
                    llm_provider=llm_provider or DEFAULT_LLM_PROVIDER,
                    embedder=embedder or DEFAULT_EMBEDDER,
                    device=device,
                    batch_size=batch_size if batch_size is not None else DEFAULT_BATCH_SIZE,
                    embedder_api_key=embedder_api_key or "EMPTY",
                    embedder_base_url=embedder_base_url,
                    embedder_model=embedder_model,
                    embedder_max_tokens=embedder_max_tokens if embedder_max_tokens is not None else 256,
                    llm_max_tokens=llm_max_tokens if llm_max_tokens is not None else 500,
                    llm_temperature=llm_temperature if llm_temperature is not None else 0.0,
                    llm_max_retries=llm_max_retries if llm_max_retries is not None else 3,
                    llm_retry_wait_sec=llm_retry_wait_sec if llm_retry_wait_sec is not None else 2.0,
                    llm_context_window=llm_context_window if llm_context_window is not None else 20000,
                    llm_prompt_token_buffer=llm_prompt_token_buffer if llm_prompt_token_buffer is not None else 128,
                    llm_use_qwen_thinking_control=bool(llm_use_qwen_thinking_control),
                    retrieve_top_ks=retrieve_top_ks,
                    mode=mode or DEFAULT_MODE,
                    mem_threshold=mem_threshold if mem_threshold is not None else 30,
                    n_components=n_components if n_components is not None else 2,
                    num_seednodes=num_seednodes if num_seednodes is not None else 15,
                    damping=damping if damping is not None else 0.1,
                    router_temp=router_temp if router_temp is not None else 0.2,
                    tracker=tracker,
                    record_latency=bool(record_latency),
                )
        finally:
            tracker.restore()
            tracker.save_to_json()

        for top_k in retrieve_top_ks:
            top_k_sample_result = {
                "sample_id": sample_result["sample_id"],
                "qa": sample_result["qa_by_top_k"].get(top_k, []),
            }
            write_json(cache_files[top_k], top_k_sample_result)
            results_by_top_k[top_k][sample_id] = top_k_sample_result
            aggregated_by_top_k[top_k] = aggregate_results(
                top_k_paths[top_k],
                selected_samples,
                top_k_cache_dirs[top_k],
                results_by_top_k[top_k],
            )

    latency_summary = {"method": "memgas", "top_k": {}}
    for top_k in retrieve_top_ks:
        latencies = []
        for sample_result in aggregated_by_top_k[top_k]:
            for qa in sample_result.get("qa", []):
                latency = qa.get("retrieval_latency_seconds")
                if latency is not None:
                    latencies.append(float(latency))
        average_latency_seconds = sum(latencies) / len(latencies) if latencies else 0.0
        total_latency_seconds = sum(latencies)
        latency_summary["top_k"][str(top_k)] = {
            "query_count": len(latencies),
            "average_retrieval_latency_seconds": average_latency_seconds,
            "average_retrieval_latency_ms": average_latency_seconds * 1000.0,
            "total_retrieval_latency_seconds": total_latency_seconds,
            "total_retrieval_latency_ms": total_latency_seconds * 1000.0,
        }
        write_json(os.path.join(os.path.dirname(top_k_paths[top_k]), "latency_summary.json"), latency_summary["top_k"][str(top_k)])
    write_json(os.path.join(output_dir, "latency_summary.json"), latency_summary)

    aggregate_token_stats(token_file, selected_samples, token_dir, retrieve_top_ks)

    if auto_eval:
        from eval import main as eval_main
        if not eval_result_path:
            raise ValueError(
                "MemGAS auto_eval requires eval_result_path pointing to separately "
                "simplified predictions; raw generation is saved as result_raw.json"
            )
        embedding_model = eval_embedding_model or "Qwen3-Embedding-0.6B"
        dataset_name = eval_dataset or "loco"
        dataset_dir = "LONGMEMEVAL" if dataset_name == "lme" else "LOCOMO"
        result_root = os.path.join(CODE_DIR, "Result", dataset_dir, "memgas")
        for top_k in retrieve_top_ks:
            version = os.path.relpath(os.path.dirname(top_k_paths[top_k]), result_root)
            print(f"Running evaluation for memgas top_k={top_k}, version={version}")
            result_path = str(eval_result_path).format(top_k=top_k)
            eval_main(
                dataset_name,
                "memgas",
                version,
                embedding_model,
                result_path=result_path,
            )

    return aggregated_by_top_k
