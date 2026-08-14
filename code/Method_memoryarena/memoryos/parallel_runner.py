import json
import multiprocessing as mp
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

from .main import (
    build_topk_output_paths,
    load_existing_results_by_k,
    parse_retrieve_top_ks,
    run_memoryos,
    save_results,
)


def _write_manifest(output_path, output_paths, results_by_k, retrieve_top_ks):
    manifest = {"retrieve_top_ks": retrieve_top_ks, "outputs": {}}
    for top_k in retrieve_top_ks:
        latencies = [
            qa.get("retrieval_latency_ms")
            for sample in results_by_k[top_k]
            for qa in sample.get("qa", [])
            if isinstance(qa.get("retrieval_latency_ms"), (int, float))
        ]
        manifest["outputs"][str(top_k)] = {
            "result_file": output_paths[top_k],
            "sample_count": len(results_by_k[top_k]),
            "qa_count": sum(len(sample.get("qa", [])) for sample in results_by_k[top_k]),
            "average_retrieval_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        }
    manifest_root = output_path if not output_path.endswith(".json") else os.path.dirname(output_path)
    save_results(os.path.join(manifest_root, "memoryos_run_manifest.json"), manifest)


def _process_sample(sample_index, total_samples, sample, worker_config):
    sample_id = sample.get("sample_id", f"sample_{sample_index}")
    safe_sample_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(sample_id))
    work_root = os.path.join(worker_config["output_root"], ".parallel_work")
    work_dir = os.path.join(work_root, f"{sample_index:06d}_{safe_sample_id}")
    dataset_path = os.path.join(work_dir, "sample.json")
    sample_output_path = os.path.join(work_dir, "output")
    os.makedirs(work_dir, exist_ok=True)
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump([sample], f, ensure_ascii=False)

    track_tokens = bool(worker_config.get("track_tokens", False))
    worker_token_file = worker_config["token_file"]
    if track_tokens:
        token_dir = f"{worker_token_file}.workers"
        os.makedirs(token_dir, exist_ok=True)
        worker_token_file = os.path.join(
            token_dir, f"{sample_index:06d}_{safe_sample_id}.json"
        )

    print(f"[parallel] Processing sample {sample_index + 1}/{total_samples}: {sample_id}", flush=True)
    results_by_k = run_memoryos(
        dataset_path=dataset_path,
        output_path=sample_output_path,
        memory_path=worker_config["memory_path"],
        llm_model=worker_config["llm_model"],
        llm_api_key=worker_config["llm_api_key"],
        llm_base_url=worker_config["llm_base_url"],
        embedding_model_name=worker_config["embedding_model_name"],
        embedding_api_key=worker_config.get("embedding_api_key", "EMPTY"),
        embedding_base_url=worker_config.get("embedding_base_url"),
        token_file=worker_token_file,
        retrieve_top_ks=worker_config["retrieve_top_ks"],
        short_term_capacity=worker_config["short_term_capacity"],
        qa_concurrency=worker_config["qa_concurrency"],
        track_tokens=track_tokens,
        fast_index=worker_config["fast_index"],
        update_profiles=worker_config["update_profiles"],
        use_retrieval_keywords=worker_config["use_retrieval_keywords"],
        memory_granularity=worker_config.get("memory_granularity", "message"),
        segment_threshold=worker_config.get("segment_threshold", 0.5),
        segment_max_messages=worker_config.get("segment_max_messages", 0),
    )

    sample_results = {}
    for top_k in worker_config["retrieve_top_ks"]:
        top_k_results = results_by_k.get(top_k, [])
        if len(top_k_results) != 1 or top_k_results[0].get("sample_id") != sample_id:
            raise RuntimeError(f"Unexpected top_k={top_k} result for sample {sample_id}")
        sample_results[top_k] = top_k_results[0]

    shutil.rmtree(work_dir, ignore_errors=True)
    return sample_index, sample_id, sample_results


def run_memoryos_parallel(
    dataset_path,
    output_path,
    memory_path,
    llm_model,
    llm_api_key,
    llm_base_url,
    embedding_model_name,
    token_file,
    embedding_api_key="EMPTY",
    embedding_base_url=None,
    retrieve_top_ks=None,
    short_term_capacity=1,
    qa_concurrency=1,
    sample_concurrency=1,
    track_tokens=False,
    fast_index=False,
    update_profiles=True,
    use_retrieval_keywords=True,
    memory_granularity="message",
    segment_threshold=0.5,
    segment_max_messages=0,
):
    sample_concurrency = max(1, int(sample_concurrency or 1))
    if sample_concurrency == 1:
        return run_memoryos(
            dataset_path=dataset_path,
            output_path=output_path,
            memory_path=memory_path,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_model_name=embedding_model_name,
            token_file=token_file,
            retrieve_top_ks=retrieve_top_ks,
            short_term_capacity=short_term_capacity,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            qa_concurrency=qa_concurrency,
            track_tokens=track_tokens,
            fast_index=fast_index,
            update_profiles=update_profiles,
            use_retrieval_keywords=use_retrieval_keywords,
            memory_granularity=memory_granularity,
            segment_threshold=segment_threshold,
            segment_max_messages=segment_max_messages,
        )
    if track_tokens:
        raise ValueError("Token tracking is not supported with sample_concurrency > 1")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    retrieve_top_ks = parse_retrieve_top_ks(retrieve_top_ks)
    output_paths = build_topk_output_paths(output_path, retrieve_top_ks)
    for output_file in output_paths.values():
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    os.makedirs(memory_path, exist_ok=True)

    results_by_k, processed_by_k = load_existing_results_by_k(output_paths)
    pending = [
        (index, sample)
        for index, sample in enumerate(dataset)
        if not all(sample.get("sample_id") in processed_by_k[top_k] for top_k in retrieve_top_ks)
    ]
    print(
        f"Starting sample-level multiprocessing with {sample_concurrency} workers: "
        f"{len(dataset) - len(pending)} completed, {len(pending)} pending.",
        flush=True,
    )
    if not pending:
        _write_manifest(output_path, output_paths, results_by_k, retrieve_top_ks)
        return results_by_k

    output_root = output_path if not output_path.endswith(".json") else os.path.dirname(output_path)
    worker_config = {
        "output_root": output_root,
        "memory_path": memory_path,
        "llm_model": llm_model,
        "llm_api_key": llm_api_key,
        "llm_base_url": llm_base_url,
        "embedding_model_name": embedding_model_name,
        "embedding_api_key": embedding_api_key,
        "embedding_base_url": embedding_base_url,
        "token_file": token_file,
        "retrieve_top_ks": retrieve_top_ks,
        "short_term_capacity": short_term_capacity,
        "qa_concurrency": qa_concurrency,
        "fast_index": fast_index,
        "update_profiles": update_profiles,
        "use_retrieval_keywords": use_retrieval_keywords,
        "memory_granularity": memory_granularity,
        "segment_threshold": segment_threshold,
        "segment_max_messages": segment_max_messages,
    }
    dataset_order = {sample.get("sample_id"): index for index, sample in enumerate(dataset)}
    failures = []
    mp_context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=sample_concurrency, mp_context=mp_context) as executor:
        future_meta = {}
        for sample_index, sample in pending:
            future = executor.submit(
                _process_sample,
                sample_index,
                len(dataset),
                sample,
                worker_config,
            )
            future_meta[future] = (sample_index, sample.get("sample_id"))

        for future in as_completed(future_meta):
            sample_index, sample_id = future_meta[future]
            try:
                _, sample_id, sample_results = future.result()
            except Exception as exc:
                failures.append((sample_index, sample_id, str(exc)))
                print(f"[parallel] Sample {sample_index + 1}/{len(dataset)} failed: {sample_id}: {exc}", flush=True)
                continue

            for top_k in retrieve_top_ks:
                if sample_id not in processed_by_k[top_k]:
                    results_by_k[top_k].append(sample_results[top_k])
                    processed_by_k[top_k].add(sample_id)
                results_by_k[top_k].sort(
                    key=lambda item: dataset_order.get(item.get("sample_id"), len(dataset))
                )
                save_results(output_paths[top_k], results_by_k[top_k])
            _write_manifest(output_path, output_paths, results_by_k, retrieve_top_ks)
            completed = min(len(processed_by_k[top_k]) for top_k in retrieve_top_ks)
            print(
                f"[parallel] Sample {sample_index + 1}/{len(dataset)} completed: {sample_id}. "
                f"Persisted progress {completed}/{len(dataset)}.",
                flush=True,
            )

    if failures:
        details = "; ".join(f"{index + 1}:{sample_id}: {error}" for index, sample_id, error in failures)
        raise RuntimeError(f"{len(failures)} parallel samples failed: {details}")
    return results_by_k
