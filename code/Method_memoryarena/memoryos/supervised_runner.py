import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from collections import deque

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .main import (
    build_topk_output_paths,
    load_existing_results_by_k,
    parse_retrieve_top_ks,
    run_memoryos,
    save_results,
)
from .parallel_runner import _process_sample, _write_manifest


def _sample_process_entry(result_queue, sample_index, total_samples, sample, worker_config):
    try:
        sample_index, sample_id, sample_results = _process_sample(
            sample_index,
            total_samples,
            sample,
            worker_config,
        )
        result_queue.put({
            "status": "ok",
            "sample_index": sample_index,
            "sample_id": sample_id,
            "sample_results": sample_results,
        })
    except BaseException as exc:
        result_queue.put({
            "status": "error",
            "sample_index": sample_index,
            "sample_id": sample.get("sample_id"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


def _aggregate_worker_tokens(token_file):
    if not token_file:
        return
    worker_dir = f"{token_file}.workers"
    if not os.path.isdir(worker_dir):
        return

    workers = []
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    for filename in sorted(os.listdir(worker_dir)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(worker_dir, filename)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                usage = json.load(handle)
        except Exception as exc:
            print(f"Warning: could not read worker token file {path}: {exc}")
            continue
        record = {"worker_file": filename}
        for key in totals:
            value = int(usage.get(key, 0) or 0)
            record[key] = value
            totals[key] += value
        workers.append(record)

    os.makedirs(os.path.dirname(token_file) or ".", exist_ok=True)
    aggregate = {
        "name": "root",
        **totals,
        "worker_count": len(workers),
        "workers": workers,
    }
    temporary_path = f"{token_file}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, token_file)
    print(f"Aggregated token usage from {len(workers)} sample workers: {token_file}")


def run_memoryos_supervised(
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
    sample_max_retries=2,
    track_tokens=False,
    fast_index=False,
    update_profiles=True,
    use_retrieval_keywords=True,
    memory_granularity="message",
    segment_threshold=0.5,
    segment_max_messages=0,
):
    sample_concurrency = max(1, int(sample_concurrency or 1))
    sample_max_retries = max(0, int(sample_max_retries or 0))
    if sample_concurrency == 1:
        return run_memoryos(
            dataset_path=dataset_path,
            output_path=output_path,
            memory_path=memory_path,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            llm_base_url=llm_base_url,
            embedding_model_name=embedding_model_name,
            token_file=token_file,
            retrieve_top_ks=retrieve_top_ks,
            short_term_capacity=short_term_capacity,
            qa_concurrency=qa_concurrency,
            track_tokens=track_tokens,
            fast_index=fast_index,
            update_profiles=update_profiles,
            use_retrieval_keywords=use_retrieval_keywords,
            memory_granularity=memory_granularity,
            segment_threshold=segment_threshold,
            segment_max_messages=segment_max_messages,
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    retrieve_top_ks = parse_retrieve_top_ks(retrieve_top_ks)
    output_paths = build_topk_output_paths(output_path, retrieve_top_ks)
    for output_file in output_paths.values():
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    os.makedirs(memory_path, exist_ok=True)

    results_by_k, processed_by_k = load_existing_results_by_k(output_paths)
    pending = deque(
        (index, sample, 0)
        for index, sample in enumerate(dataset)
        if not all(sample.get("sample_id") in processed_by_k[top_k] for top_k in retrieve_top_ks)
    )
    print(
        f"Starting supervised sample multiprocessing with {sample_concurrency} workers: "
        f"{len(dataset) - len(pending)} completed, {len(pending)} pending, "
        f"max retries={sample_max_retries}.",
        flush=True,
    )
    if not pending:
        if track_tokens:
            _aggregate_worker_tokens(token_file)
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
        "track_tokens": track_tokens,
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
    context = mp.get_context("spawn")
    active = {}
    failures = []

    def launch(sample_index, sample, attempt):
        result_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=_sample_process_entry,
            args=(result_queue, sample_index, len(dataset), sample, worker_config),
            name=f"memoryos-sample-{sample_index + 1}-attempt-{attempt + 1}",
        )
        process.start()
        active[sample_index] = {
            "process": process,
            "queue": result_queue,
            "sample": sample,
            "attempt": attempt,
        }
        print(
            f"[supervisor] Launched sample {sample_index + 1}/{len(dataset)} "
            f"(attempt {attempt + 1}/{sample_max_retries + 1}, pid={process.pid}).",
            flush=True,
        )

    def merge_success(message):
        sample_index = message["sample_index"]
        sample_id = message["sample_id"]
        sample_results = {int(top_k): value for top_k, value in message["sample_results"].items()}
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
            f"[supervisor] Sample {sample_index + 1}/{len(dataset)} completed: {sample_id}. "
            f"Persisted progress {completed}/{len(dataset)}.",
            flush=True,
        )

    try:
        while pending or active:
            while pending and len(active) < sample_concurrency:
                launch(*pending.popleft())

            finished = []
            for sample_index, job in list(active.items()):
                message = None
                try:
                    message = job["queue"].get_nowait()
                except queue.Empty:
                    pass

                process = job["process"]
                if message is not None:
                    process.join(timeout=10)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                    finished.append((sample_index, message, process.exitcode))
                elif not process.is_alive():
                    process.join()
                    try:
                        message = job["queue"].get(timeout=1)
                    except queue.Empty:
                        message = {
                            "status": "abrupt_exit",
                            "sample_index": sample_index,
                            "sample_id": job["sample"].get("sample_id"),
                            "error": f"worker exited without a result (exitcode={process.exitcode})",
                        }
                    finished.append((sample_index, message, process.exitcode))

            if not finished:
                time.sleep(1)
                continue

            for sample_index, message, exitcode in finished:
                job = active.pop(sample_index)
                job["queue"].close()
                if message["status"] == "ok":
                    merge_success(message)
                    continue

                error = message.get("error", f"worker failed with exitcode={exitcode}")
                detail = message.get("traceback", "")
                attempt = job["attempt"]
                sample_id = job["sample"].get("sample_id")
                print(
                    f"[supervisor] Sample {sample_index + 1}/{len(dataset)} failed "
                    f"on attempt {attempt + 1} with exitcode={exitcode}: {error}",
                    flush=True,
                )
                if detail:
                    print(detail, flush=True)
                if attempt < sample_max_retries:
                    pending.appendleft((sample_index, job["sample"], attempt + 1))
                    print(
                        f"[supervisor] Re-queued sample {sample_index + 1}/{len(dataset)}.",
                        flush=True,
                    )
                else:
                    failures.append((sample_index, sample_id, error))
    finally:
        for job in active.values():
            process = job["process"]
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            job["queue"].close()

    if failures:
        details = "; ".join(
            f"{sample_index + 1}:{sample_id}: {error}"
            for sample_index, sample_id, error in failures
        )
        raise RuntimeError(f"{len(failures)} samples exhausted retries: {details}")
    if track_tokens:
        _aggregate_worker_tokens(token_file)
    return results_by_k
