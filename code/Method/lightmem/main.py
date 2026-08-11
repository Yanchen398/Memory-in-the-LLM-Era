import json
import os
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .prompts import ANSWER_PROMPT, METADATA_GENERATE_PROMPT_locomo


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LIGHTMEM_SRC_DIR = os.path.join(CURRENT_DIR, "src")
if LIGHTMEM_SRC_DIR not in sys.path:
    sys.path.insert(0, LIGHTMEM_SRC_DIR)

CODE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
DEFAULT_OUTPUT_ROOT = os.path.abspath(os.path.join(CODE_DIR, "Result/LOCOMO"))

DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"
DEFAULT_LLM_API_KEY = "empty"
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLMLINGUA_MODEL_NAME = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
DEFAULT_RETRIEVE_K = 10
NO_THINKING_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
DEFAULT_BATCH_SIZE = 64
DEFAULT_RATIO = 1.0


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
        return os.path.join(DEFAULT_OUTPUT_ROOT, "lightmem", "default", "result.json")
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


def build_cache_file(cache_dir: str, dataset_index: int, sample_id: str) -> str:
    return os.path.join(cache_dir, f"{dataset_index:05d}_{safe_name(sample_id)}.json")


def collect_existing_results(output_path: str, selected_samples: List[Tuple[int, Dict]], cache_dir: str) -> Dict[str, Dict]:
    results_by_id: Dict[str, Dict] = {}
    existing_output = read_json(output_path, [])
    if isinstance(existing_output, list):
        for item in existing_output:
            sample_id = item.get("sample_id")
            if sample_id is not None:
                results_by_id[str(sample_id)] = item

    for dataset_index, sample in selected_samples:
        sample_id = str(sample.get("sample_id", dataset_index))
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
        sample_id = str(sample.get("sample_id", dataset_index))
        cache_file = build_cache_file(cache_dir, dataset_index, sample_id)
        if os.path.exists(cache_file):
            results_by_id[sample_id] = read_json(cache_file, {})
        if sample_id in results_by_id:
            aggregated.append(results_by_id[sample_id])
    write_json(output_path, aggregated)
    return aggregated


def parse_locomo_timestamp(timestamp_str: str) -> str:
    timestamp_str = str(timestamp_str or "").strip("() ")
    try:
        dt = datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return timestamp_str


def format_turn_text(turn: Dict) -> str:
    text = str(turn.get("text", ""))
    caption = str(turn.get("blip_caption", "")).strip()
    if caption:
        return f"{text} (image description: {caption})"
    return text


def extract_locomo_turn_messages(conversation: Dict) -> Tuple[List[Tuple[List[Dict], bool]], str, str]:
    speaker_a = conversation.get("speaker_a", "Speaker_A")
    speaker_b = conversation.get("speaker_b", "Speaker_B")
    session_keys = sorted(
        key for key in conversation.keys() if key.startswith("session_") and not key.endswith("_date_time")
    )

    batches: List[Tuple[List[Dict], bool]] = []
    for session_index, session_key in enumerate(session_keys):
        timestamp = parse_locomo_timestamp(conversation.get(f"{session_key}_date_time", ""))
        turns = conversation.get(session_key, []) or []
        for turn_index, turn in enumerate(turns):
            speaker_name = turn.get("speaker", "")
            speaker_id = "speaker_a" if speaker_name == speaker_a else "speaker_b"
            content = format_turn_text(turn)
            messages = [
                {
                    "role": "user",
                    "content": content,
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                    "time_stamp": timestamp,
                },
                {
                    "role": "assistant",
                    "content": "",
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                    "time_stamp": timestamp,
                },
            ]
            is_last = session_index == len(session_keys) - 1 and turn_index == len(turns) - 1
            batches.append((messages, is_last))
    return batches, speaker_a, speaker_b


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
    pre_compress: bool,
    llmlingua_model_path: Optional[str],
    compression_rate: float,
    topic_segment: bool,
    metadata_generate: bool,
    text_summary: bool,
    extraction_mode: str,
) -> Dict:
    llmlingua_model_name = llmlingua_model_path or DEFAULT_LLMLINGUA_MODEL_NAME
    memory_manager_configs = {
        "model": llm_model,
        "api_key": llm_api_key,
        "max_tokens": 4096,
        "extra_body": NO_THINKING_EXTRA_BODY,
    }
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
        config["topic_segmenter"] = {
            "model_name": "llmlingua-2",
            "configs": {
                "model_name": llmlingua_model_name,
                "device_map": embedding_device,
                "use_llmlingua2": True,
            },
        }
    else:
        config["pre_compressor"] = {"model_name": "entropy_compress"}
        config["topic_segmenter"] = {
            "model_name": "llmlingua-2",
            "configs": {
                "model_name": llmlingua_model_name,
                "device_map": embedding_device,
                "use_llmlingua2": True,
            },
        }

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


def aggregate_token_stats(token_file: str, sample_stats: List[Dict]):
    payload = {
        "method": "lightmem",
        "sample_count": len(sample_stats),
        "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in sample_stats),
        "completion_tokens": sum(item.get("completion_tokens", 0) for item in sample_stats),
        "total_tokens": sum(item.get("total_tokens", 0) for item in sample_stats),
        "samples": sample_stats,
    }
    write_json(token_file, payload)


def generate_answer(client, model: str, question: str, retrieved: List[str], speaker_a: str, speaker_b: str) -> Tuple[str, Dict[str, int]]:
    context_a = "\n\n".join(retrieved) if retrieved else "No memories available."
    prompt = ANSWER_PROMPT.format(
        speaker_1_name=speaker_a,
        speaker_1_memories=context_a,
        speaker_2_name=speaker_b,
        speaker_2_memories="See retrieved memories above.",
        question=question,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        extra_body=NO_THINKING_EXTRA_BODY,
    )
    usage = getattr(response, "usage", None)
    token_usage = {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }
    content = response.choices[0].message.content if response.choices else ""
    return (content or "").strip(), token_usage


def process_sample(
    sample: Dict,
    collection_name: str,
    sample_storage_dir: str,
    sample_log_dir: str,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
    llm_provider: str,
    embedding_model_name: str,
    embedding_dim: int,
    embedding_device: str,
    retrieve_k: int,
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
) -> Tuple[Dict, Dict]:
    try:
        from lightmem.memory.lightmem import LightMemory
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "LightMem requires its runtime dependencies, including `torch`, `transformers`, "
            "`sentence-transformers`, `llmlingua`, `qdrant-client`, `pydantic`, and `openai`."
        ) from exc

    if os.path.exists(sample_storage_dir):
        shutil.rmtree(sample_storage_dir)
    os.makedirs(sample_storage_dir, exist_ok=True)
    os.makedirs(sample_log_dir, exist_ok=True)

    config = build_lightmem_config(
        collection_name=collection_name,
        storage_dir=sample_storage_dir,
        log_dir=sample_log_dir,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_provider=llm_provider,
        embedding_model_name=embedding_model_name,
        embedding_dim=embedding_dim,
        embedding_device=embedding_device,
        pre_compress=pre_compress,
        llmlingua_model_path=llmlingua_model_path,
        compression_rate=compression_rate,
        topic_segment=topic_segment,
        metadata_generate=metadata_generate,
        text_summary=text_summary,
        extraction_mode=extraction_mode,
    )
    lightmem = LightMemory.from_config(config)

    batches, speaker_a, speaker_b = extract_locomo_turn_messages(sample.get("conversation", {}) or {})
    for messages, is_last in batches:
        lightmem.add_memory(
            messages=messages,
            METADATA_GENERATE_PROMPT=METADATA_GENERATE_PROMPT_locomo,
            force_segment=is_last,
            force_extract=is_last,
        )

    if enable_update:
        lightmem.construct_update_queue_all_entries(top_k=update_top_k, keep_top_n=update_keep_top_n, max_workers=update_workers)
        lightmem.offline_update_all_entries(score_threshold=update_score_threshold, max_workers=update_workers)

    if enable_summary:
        lightmem.summarize(
            retrieval_scope="global",
            time_window=summary_time_window,
            top_k_seeds=summary_top_k_seeds,
            process_all=True,
        )

    answer_client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
    qa_results = []
    answer_token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for qa in sample.get("qa", []):
        question = qa.get("question", "")
        answer = qa.get("answer") or qa.get("adversarial_answer", "")
        category = qa.get("category")
        retrieved = lightmem.retrieve(question, limit=retrieve_k)
        response, token_usage = generate_answer(answer_client, llm_model, question, retrieved, speaker_a, speaker_b)
        for key in answer_token_totals:
            answer_token_totals[key] += token_usage[key]
        qa_results.append(
            {
                "question": question,
                "answer": answer,
                "category": category,
                "response": response,
                "retrieved": retrieved,
            }
        )

    memory_tokens = summarize_token_stats(lightmem.get_token_statistics())
    sample_tokens = {
        "prompt_tokens": memory_tokens["prompt_tokens"] + answer_token_totals["prompt_tokens"],
        "completion_tokens": memory_tokens["completion_tokens"] + answer_token_totals["completion_tokens"],
        "total_tokens": memory_tokens["total_tokens"] + answer_token_totals["total_tokens"],
        "memory_tokens": memory_tokens,
        "answer_tokens": answer_token_totals,
    }
    return {"sample_id": str(sample.get("sample_id", collection_name)), "qa": qa_results}, sample_tokens


def run_lightmem(
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
    retrieve_k=DEFAULT_RETRIEVE_K,
    retrieve_top_ks=None,
    ratio=DEFAULT_RATIO,
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
    config_path=None,
    config_overrides=None,
):
    from .official import load_config_extras, run_lightmem_official

    extras = load_config_extras(config_path)
    if config_overrides:
        extras.update(config_overrides)
    if not extras.get("legacy_flow", False):
        return run_lightmem_official(
            dataset_path=dataset_path,
            output_path=output_path,
            memory_path=memory_path,
            token_file=token_file,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            llm_provider=llm_provider,
            embedding_model_name=embedding_model_name,
            embedding_dim=embedding_dim,
            embedding_device=embedding_device,
            retrieve_k=retrieve_k,
            retrieve_top_ks=retrieve_top_ks,
            ratio=ratio,
            start_idx=start_idx,
            end_idx=end_idx,
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
            auto_eval=auto_eval,
            eval_embedding_model=eval_embedding_model,
            config_path=config_path,
            config_overrides=config_overrides,
        )

    base_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()

    dataset_path = resolve_path(dataset_path, base_dir)
    if not dataset_path:
        raise ValueError("LightMem requires 'dataset_path' in the config or CLI arguments.")

    output_path = normalize_output_path(resolve_path(output_path, base_dir))
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    memory_root = resolve_path(memory_path, base_dir) or os.path.join(output_dir, "_lightmem_runtime")
    cache_dir = os.path.join(output_dir, "_lightmem_cache")
    log_dir = os.path.join(output_dir, "_lightmem_logs")
    os.makedirs(memory_root, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    token_file = resolve_path(token_file, base_dir) or os.path.join(output_dir, "token_tracker.json")

    all_samples = load_dataset(dataset_path)
    selected_samples = apply_sample_slice(
        all_samples,
        start_idx=start_idx if start_idx is not None else 0,
        end_idx=end_idx,
        ratio=ratio if ratio is not None else DEFAULT_RATIO,
    )

    results_by_id = collect_existing_results(output_path, selected_samples, cache_dir)
    aggregated_results = aggregate_results(output_path, selected_samples, cache_dir, results_by_id)
    token_stats: List[Dict] = []

    for ordinal, (dataset_index, sample) in enumerate(selected_samples, start=1):
        sample_id = str(sample.get("sample_id", dataset_index))
        cache_file = build_cache_file(cache_dir, dataset_index, sample_id)
        if os.path.exists(cache_file):
            print(f"[{ordinal}/{len(selected_samples)}] Skip completed sample: {sample_id}")
            continue
        if sample_id in results_by_id:
            write_json(cache_file, results_by_id[sample_id])
            print(f"[{ordinal}/{len(selected_samples)}] Reused aggregated sample: {sample_id}")
            continue

        collection_name = safe_name(sample_id)
        sample_storage_dir = os.path.join(memory_root, f"{dataset_index:05d}_{collection_name}")
        sample_log_dir = os.path.join(log_dir, f"{dataset_index:05d}_{collection_name}")

        sample_result, sample_tokens = process_sample(
            sample=sample,
            collection_name=collection_name,
            sample_storage_dir=sample_storage_dir,
            sample_log_dir=sample_log_dir,
            llm_model=llm_model or DEFAULT_LLM_MODEL,
            llm_api_key=llm_api_key or DEFAULT_LLM_API_KEY,
            llm_base_url=llm_base_url or DEFAULT_LLM_BASE_URL,
            llm_provider=llm_provider or "openai",
            embedding_model_name=embedding_model_name or DEFAULT_EMBEDDING_MODEL_NAME,
            embedding_dim=embedding_dim or 384,
            embedding_device=embedding_device or "cpu",
            retrieve_k=retrieve_k if retrieve_k is not None else DEFAULT_RETRIEVE_K,
            pre_compress=bool(pre_compress),
            llmlingua_model_path=resolve_path(llmlingua_model_path, base_dir) if llmlingua_model_path else None,
            compression_rate=compression_rate if compression_rate is not None else 0.6,
            topic_segment=bool(topic_segment),
            metadata_generate=bool(metadata_generate),
            text_summary=bool(text_summary),
            extraction_mode=extraction_mode or "flat",
            enable_update=bool(enable_update),
            update_top_k=update_top_k if update_top_k is not None else 20,
            update_keep_top_n=update_keep_top_n if update_keep_top_n is not None else 10,
            update_score_threshold=update_score_threshold if update_score_threshold is not None else 0.9,
            update_workers=update_workers if update_workers is not None else 5,
            enable_summary=bool(enable_summary),
            summary_time_window=summary_time_window if summary_time_window is not None else 3600,
            summary_top_k_seeds=summary_top_k_seeds if summary_top_k_seeds is not None else 15,
        )
        write_json(cache_file, sample_result)
        results_by_id[sample_id] = sample_result
        token_stats.append({"sample_id": sample_id, **sample_tokens})
        aggregated_results = aggregate_results(output_path, selected_samples, cache_dir, results_by_id)

    aggregate_token_stats(token_file, token_stats)
    return aggregated_results
