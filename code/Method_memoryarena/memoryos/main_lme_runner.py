import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import time
import traceback
from collections import deque

from ..dataset_hygiene import (
    natural_session_keys,
    pair_session_turns,
    raw_result_path,
    resolve_required_endpoint,
)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_RETRIEVAL_TOP_K = 10


def _save_json(path, value):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def _configure_library_runtime(
    embedding_model_name,
    embedding_api_key,
    embedding_base_url,
    llm_model,
):
    import memoryos.long_term as library_long_term
    import memoryos.mid_term as library_mid_term
    import memoryos.prompts as library_prompts
    import memoryos.updater as library_updater
    import memoryos.utils as library_utils
    import numpy as np
    from openai import OpenAI

    from .utils import (
        OpenAIClient as ProjectOpenAIClient,
        normalize_multi_summaries,
    )

    embedding_client = OpenAI(
        api_key=embedding_api_key,
        base_url=embedding_base_url,
    )
    original_extract_keywords = library_utils.llm_extract_keywords

    def configured_get_embedding(text, model_name=None):
        last_error = None
        for attempt in range(3):
            try:
                response = embedding_client.embeddings.create(
                    model=embedding_model_name,
                    input=str(text),
                    extra_body={"truncate_prompt_tokens": 256},
                )
                return np.asarray(response.data[0].embedding, dtype=np.float32)
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1 + attempt)
        raise RuntimeError(f"Embedding API failed after 3 attempts: {last_error}")

    def configured_extract_keywords(text, client, model=None):
        return original_extract_keywords(text, client, model=llm_model)

    def configured_multi_summary(text, client, model=None):
        messages = [
            {"role": "system", "content": library_prompts.MULTI_SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": library_prompts.MULTI_SUMMARY_USER_PROMPT.format(text=text),
            },
        ]
        response_text = client.chat_completion(model=llm_model, messages=messages)
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            summaries = normalize_multi_summaries(json.loads(cleaned))
        except Exception:
            print(f"[main_lme] Could not parse multi-summary JSON: {response_text}")
            summaries = []
        return {"input": text, "summaries": summaries}

    library_utils.OpenAIClient.chat_completion = ProjectOpenAIClient.chat_completion
    library_utils.get_embedding = configured_get_embedding
    library_utils.llm_extract_keywords = configured_extract_keywords
    library_utils.gpt_generate_multi_summary = configured_multi_summary
    library_mid_term.get_embedding = configured_get_embedding
    library_mid_term.llm_extract_keywords = configured_extract_keywords
    library_long_term.get_embedding = configured_get_embedding
    library_updater.llm_extract_keywords = configured_extract_keywords
    library_updater.gpt_generate_multi_summary = configured_multi_summary


def _collect_retrieved(retrieval_result):
    retrieved = []
    if not retrieval_result:
        return retrieved
    for page in retrieval_result.get("retrieved_pages", []):
        if page.get("user_input"):
            retrieved.append(page["user_input"])
        if page.get("agent_response"):
            retrieved.append(page["agent_response"])
    for key in ("retrieved_user_knowledge", "retrieved_assistant_knowledge"):
        for item in retrieval_result.get(key, []):
            if item.get("knowledge"):
                retrieved.append(item["knowledge"])
    return retrieved


def _process_sample(sample_index, total_samples, sample, worker_config):
    from memoryos import Memoryos

    _configure_library_runtime(
        worker_config["embedding_model_name"],
        worker_config["embedding_api_key"],
        worker_config["embedding_base_url"],
        worker_config["llm_model"],
    )

    sample_id = sample.get("sample_id", f"sample_{sample_index}")
    safe_sample_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(sample_id))
    sample_memory_path = os.path.join(
        worker_config["memory_path"],
        f"{sample_index:06d}_{safe_sample_id}",
    )
    shutil.rmtree(sample_memory_path, ignore_errors=True)
    os.makedirs(sample_memory_path, exist_ok=True)

    print(
        f"[main_lme] Processing sample {sample_index + 1}/{total_samples}: {sample_id}",
        flush=True,
    )
    conversation = sample["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    memo = Memoryos(
        user_id=f"user_{sample_id}",
        openai_api_key=worker_config["llm_api_key"],
        openai_base_url=worker_config["llm_base_url"],
        data_storage_path=sample_memory_path,
        llm_model=worker_config["llm_model"],
        assistant_id=f"assistant_{sample_id}",
        short_term_capacity=7,
        mid_term_heat_threshold=5,
        retrieval_queue_capacity=DEFAULT_RETRIEVAL_TOP_K,
    )

    for key in natural_session_keys(conversation):
        print(f"[main_lme] {sample_id}: indexing {key}", flush=True)
        for exchange in pair_session_turns(conversation, key, speaker_a, speaker_b):
            memo.add_memory(
                user_input=exchange["query"],
                agent_response=exchange["response"],
            )

    qa_results = []
    for qa_index, qa in enumerate(sample.get("qa", [])):
        question = qa["question"]
        original_answer = qa.get("answer", "") or qa.get("adversarial_answer", "")
        captured = {}
        original_retrieve = memo.retriever.retrieve_context

        def timed_retrieve(*args, **kwargs):
            retrieval_start = time.perf_counter()
            result = original_retrieve(*args, **kwargs)
            captured["latency_ms"] = (time.perf_counter() - retrieval_start) * 1000.0
            captured["result"] = result
            return result

        memo.retriever.retrieve_context = timed_retrieve
        try:
            system_answer = memo.get_response(query=question)
        finally:
            memo.retriever.retrieve_context = original_retrieve

        if system_answer.startswith("Error: Could not get response from LLM"):
            raise RuntimeError(f"Library LLM call failed for sample {sample_id}, QA {qa_index + 1}")

        qa_results.append({
            "question": question,
            "answer": original_answer,
            "category": qa.get("category"),
            "response": system_answer,
            "retrieved": _collect_retrieved(captured.get("result")),
            "retrieval_top_k": DEFAULT_RETRIEVAL_TOP_K,
            "retrieval_latency_ms": captured.get("latency_ms", 0.0),
        })

    return sample_index, {
        "sample_id": sample_id,
        "qa": qa_results,
    }


def _worker_entry(result_queue, sample_index, total_samples, sample, worker_config):
    try:
        sample_index, result = _process_sample(
            sample_index,
            total_samples,
            sample,
            worker_config,
        )
        result_queue.put({
            "status": "ok",
            "sample_index": sample_index,
            "result": result,
        })
    except BaseException as exc:
        result_queue.put({
            "status": "error",
            "sample_index": sample_index,
            "sample_id": sample.get("sample_id"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


def _write_manifest(output_path, result_file, results):
    latencies = [
        qa.get("retrieval_latency_ms")
        for sample in results
        for qa in sample.get("qa", [])
        if isinstance(qa.get("retrieval_latency_ms"), (int, float))
    ]
    manifest = {
        "backend": "memoryos_library_main_lme",
        "retrieve_top_ks": [DEFAULT_RETRIEVAL_TOP_K],
        "outputs": {
            str(DEFAULT_RETRIEVAL_TOP_K): {
                "result_file": result_file,
                "sample_count": len(results),
                "qa_count": sum(len(sample.get("qa", [])) for sample in results),
                "average_retrieval_latency_ms": (
                    sum(latencies) / len(latencies) if latencies else 0.0
                ),
            }
        },
    }
    _save_json(os.path.join(output_path, "memoryos_run_manifest.json"), manifest)


def run_memoryos(
    dataset_path,
    output_path,
    memory_path,
    llm_model,
    llm_api_key,
    llm_base_url,
    embedding_model_name,
    embedding_api_key="EMPTY",
    embedding_base_url=None,
    sample_concurrency=16,
    sample_max_retries=2,
):
    sample_concurrency = max(1, int(sample_concurrency or 1))
    sample_max_retries = max(0, int(sample_max_retries or 0))
    embedding_base_url = resolve_required_endpoint(embedding_base_url)
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    result_file = raw_result_path(
        os.path.join(output_path, f"top_k_{DEFAULT_RETRIEVAL_TOP_K}")
    )
    results = []
    if os.path.exists(result_file):
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                results = existing
        except Exception as exc:
            print(f"[main_lme] Failed to load existing results: {exc}", flush=True)

    processed = {
        result.get("sample_id")
        for result in results
        if isinstance(result, dict) and result.get("sample_id")
    }
    pending = deque(
        (index, sample, 0)
        for index, sample in enumerate(dataset)
        if sample.get("sample_id") not in processed
    )
    print(
        f"[main_lme] Starting supervised library run with {sample_concurrency} workers: "
        f"{len(processed)} completed, {len(pending)} pending, "
        f"max retries={sample_max_retries}.",
        flush=True,
    )
    if not pending:
        _write_manifest(output_path, result_file, results)
        return results

    worker_config = {
        "memory_path": memory_path,
        "llm_model": llm_model,
        "llm_api_key": llm_api_key,
        "llm_base_url": llm_base_url,
        "embedding_model_name": embedding_model_name,
        "embedding_api_key": embedding_api_key,
        "embedding_base_url": embedding_base_url,
    }
    dataset_order = {sample.get("sample_id"): index for index, sample in enumerate(dataset)}
    context = mp.get_context("spawn")
    active = {}
    failures = []

    def launch(sample_index, sample, attempt):
        result_queue = context.Queue(maxsize=1)
        process = context.Process(
            target=_worker_entry,
            args=(result_queue, sample_index, len(dataset), sample, worker_config),
            name=f"main-lme-sample-{sample_index + 1}-attempt-{attempt + 1}",
        )
        process.start()
        active[sample_index] = {
            "process": process,
            "queue": result_queue,
            "sample": sample,
            "attempt": attempt,
        }
        print(
            f"[main_lme] Launched sample {sample_index + 1}/{len(dataset)} "
            f"(attempt {attempt + 1}/{sample_max_retries + 1}, pid={process.pid}).",
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
                    result = message["result"]
                    sample_id = result["sample_id"]
                    if sample_id not in processed:
                        results.append(result)
                        processed.add(sample_id)
                    results.sort(
                        key=lambda item: dataset_order.get(item.get("sample_id"), len(dataset))
                    )
                    _save_json(result_file, results)
                    _write_manifest(output_path, result_file, results)
                    print(
                        f"[main_lme] Sample {sample_index + 1}/{len(dataset)} completed: "
                        f"{sample_id}. Persisted progress {len(processed)}/{len(dataset)}.",
                        flush=True,
                    )
                    continue

                error = message.get("error", f"worker failed with exitcode={exitcode}")
                detail = message.get("traceback", "")
                attempt = job["attempt"]
                sample_id = job["sample"].get("sample_id")
                print(
                    f"[main_lme] Sample {sample_index + 1}/{len(dataset)} failed "
                    f"on attempt {attempt + 1} with exitcode={exitcode}: {error}",
                    flush=True,
                )
                if detail:
                    print(detail, flush=True)
                if attempt < sample_max_retries:
                    pending.appendleft((sample_index, job["sample"], attempt + 1))
                    print(
                        f"[main_lme] Re-queued sample {sample_index + 1}/{len(dataset)}.",
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
        raise RuntimeError(f"{len(failures)} main_lme samples exhausted retries: {details}")
    return results
