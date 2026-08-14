"""Re-query persisted SOTA-ablation graphs without rebuilding any index."""

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import time
import urllib.request
from types import SimpleNamespace

import yaml

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _save_json(path, value):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def _parse_indices(value, total_samples):
    if not value:
        return list(range(total_samples))
    indices = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        index = int(part)
        if index < 0 or index >= total_samples:
            raise ValueError(f"sample index out of range: {index}")
        if index not in indices:
            indices.append(index)
    return indices


def _sample_paths(output_dir, sample_index):
    return (
        os.path.join(output_dir, "sample_results", f"sample_{sample_index}.json"),
        os.path.join(output_dir, "token_workers", f"sample_{sample_index}.json"),
    )


def _source_paths(source_root, sample_index, sample_id):
    graph_path = os.path.join(
        source_root, "database", f"sample_{sample_index}_memory_graph.json"
    )
    memory_path = os.path.join(
        source_root, "memory", f"mem_data_batch_{sample_index}"
    )
    short_term_path = os.path.join(memory_path, str(sample_id), "short_term.json")
    return graph_path, memory_path, short_term_path


def _process_sample(
    sample_index,
    sample,
    config_dict,
    source_root,
    output_dir,
    llm_runtime,
    qa_concurrency,
    max_qa,
):
    from . import config as config_module
    from .memoryos import Memoryos
    from .token_tracker import get_token_usage, reset_token_usage, save_token_usage
    from .utils import configure_llm_runtime

    reset_token_usage()
    configure_llm_runtime(
        base_urls=config_dict["llm_base_url"],
        per_endpoint_concurrency=llm_runtime["per_endpoint_concurrency"],
        semaphores=llm_runtime["semaphores"],
        counter=llm_runtime["counter"],
        lock=llm_runtime["lock"],
    )
    config_module.globalconfig = SimpleNamespace(
        embedding_model_name=config_dict["embedding_model_name"],
        embedding_api_key=config_dict.get("embedding_api_key", "EMPTY"),
        embedding_base_url=config_dict["embedding_base_url"],
        embedding_batch_size=int(config_dict.get("embedding_batch_size", 256)),
    )

    sample_id = sample["sample_id"]
    graph_path, memory_path, short_term_path = _source_paths(
        source_root, sample_index, sample_id
    )
    for required_path in (graph_path, short_term_path):
        if not os.path.isfile(required_path):
            raise FileNotFoundError(
                f"Required persisted index state is missing: {required_path}"
            )

    granularity = str(config_dict["source_mode"]).removesuffix("_graph")
    memo = Memoryos(
        user_id=str(sample_id),
        openai_api_key=config_dict["llm_api_key"],
        openai_base_url=config_dict["llm_base_url"],
        data_storage_path=memory_path,
        llm_model=config_dict["llm_model"],
        short_term_capacity=int(config_dict.get("short_term_capacity", 7)),
        tree=None,
        segment_threshold=float(config_dict.get("segment_threshold", 0.5)),
        memory_granularity=granularity,
        mid_term_structure="graph",
        graph_path=graph_path,
        graph_options={
            "context_window": config_dict.get("graph_context_window", 3),
            "candidate_count": config_dict.get("graph_candidate_count", 10),
            "dedupe_candidate_count": config_dict.get(
                "graph_dedupe_candidate_count", 20
            ),
            "fuzzy_threshold": config_dict.get("graph_fuzzy_threshold", 0.90),
            "search_hops": config_dict.get("graph_search_hops", 1),
            "use_llm_dedup": config_dict.get("graph_use_llm_dedup", True),
            "use_edge_dedup": config_dict.get("graph_use_edge_dedup", True),
        },
        top_k_retrieve=int(config_dict.get("top_k_retrieve", 10)),
        graph_only_retrieval=True,
        graph_include_original_text=config_dict.get("include_original_text", False),
    )
    retrieval_mode = (
        "graph_plus_original"
        if config_dict.get("include_original_text", False)
        else "graph_only"
    )

    conversation = sample["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]
    qa_pairs = list(sample.get("qa", []))
    if max_qa is not None:
        qa_pairs = qa_pairs[:max(0, int(max_qa))]

    def answer(index_and_qa):
        qa_index, qa = index_and_qa
        retrieval_start = time.perf_counter()
        retrieved, response = memo.get_response(
            query=qa["question"],
            mode="split",
            speaker_a=speaker_a,
            speaker_b=speaker_b,
        )
        retrieval_and_answer_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000.0
        graph_contexts, dialogue_contexts = retrieved
        if dialogue_contexts:
            raise RuntimeError(
                "graph-only retrieval unexpectedly returned dialogue contexts"
            )
        if not response or str(response).startswith("Error:"):
            raise RuntimeError(
                f"Answer generation failed for {sample_id} QA {qa_index + 1}: "
                f"{response!r}"
            )
        original_answer = qa.get("answer", "") or qa.get(
            "adversarial_answer", ""
        )
        return qa_index, {
            "question": qa["question"],
            "answer": original_answer,
            "category": qa.get("category"),
            "response": response,
            "retrieved": graph_contexts,
            "retrieval_mode": retrieval_mode,
            "retrieval_top_k": int(config_dict.get("top_k_retrieve", 10)),
            "retrieval_and_answer_latency_ms": retrieval_and_answer_ms,
        }

    qa_results = [None] * len(qa_pairs)
    worker_count = max(1, min(int(qa_concurrency), len(qa_pairs) or 1))
    if worker_count == 1:
        for item in enumerate(qa_pairs):
            qa_index, qa_result = answer(item)
            qa_results[qa_index] = qa_result
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:
            futures = [
                executor.submit(answer, item)
                for item in enumerate(qa_pairs)
            ]
            for future in concurrent.futures.as_completed(futures):
                qa_index, qa_result = future.result()
                qa_results[qa_index] = qa_result

    sample_result = {
        "sample_id": sample_id,
        "retrieval_mode": retrieval_mode,
        "source_mode": config_dict["source_mode"],
        "qa": qa_results,
    }
    result_path, token_path = _sample_paths(output_dir, sample_index)
    usage = get_token_usage()
    usage["sample_index"] = sample_index
    usage["sample_id"] = sample_id
    usage["source_mode"] = config_dict["source_mode"]
    usage["retrieval_mode"] = retrieval_mode
    save_token_usage(token_path, usage)
    _save_json(result_path, sample_result)
    return sample_index, sample_result


def _graph_state(source_root, sample_index, sample_id):
    graph_path, _, _ = _source_paths(source_root, sample_index, sample_id)
    stat = os.stat(graph_path)
    digest = hashlib.sha256()
    with open(graph_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": graph_path,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def _healthy_endpoints(base_urls, timeout=3.0):
    healthy = []
    for base_url in base_urls:
        models_url = f"{base_url.rstrip('/')}/models"
        try:
            with urllib.request.urlopen(models_url, timeout=timeout) as response:
                if 200 <= response.status < 300:
                    healthy.append(base_url)
                    print(f"[endpoint] healthy: {base_url}", flush=True)
                    continue
        except Exception as exc:
            print(
                f"[endpoint] unavailable, excluded for this run: "
                f"{base_url} ({exc})",
                flush=True,
            )
    if not healthy:
        raise RuntimeError("No configured LLM endpoint is healthy")
    return healthy


def run_graph_only(
    config_path,
    source_mode,
    output_dir,
    source_root=None,
    sample_concurrency=None,
    qa_concurrency=None,
    sample_max_retries=2,
    sample_indices=None,
    max_qa=None,
    llm_base_url=None,
    include_original_text=False,
):
    if source_mode not in {"segment_graph", "message_graph"}:
        raise ValueError(
            "source_mode must be 'segment_graph' or 'message_graph'"
        )
    with open(config_path, "r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle)
    with open(config_dict["dataset_path"], "r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    indices = _parse_indices(sample_indices, len(dataset))
    if not indices:
        raise ValueError("No samples selected")
    ablation_root = source_root or config_dict.get("ablation_root")
    if not ablation_root:
        raise ValueError(
            "source_root is required; pass it explicitly or set ablation_root in the config"
        )
    source_root = os.path.join(os.path.abspath(ablation_root), source_mode)
    output_dir = os.path.abspath(output_dir)
    if os.path.abspath(source_root) == output_dir:
        raise ValueError("output_dir must differ from the source index directory")
    os.makedirs(output_dir, exist_ok=True)

    config_dict["source_mode"] = source_mode
    config_dict["graph_only_retrieval"] = True
    config_dict["top_k_retrieve"] = 10
    config_dict["include_original_text"] = bool(include_original_text)
    retrieval_mode = (
        "graph_plus_original"
        if include_original_text
        else "graph_only"
    )
    base_urls = llm_base_url or config_dict["llm_base_url"]
    if not isinstance(base_urls, (list, tuple)):
        base_urls = [
            item.strip()
            for item in str(base_urls).split(",")
            if item.strip()
        ]
    base_urls = _healthy_endpoints(base_urls)
    config_dict["llm_base_url"] = list(base_urls)

    sample_concurrency = max(
        1,
        min(
            int(sample_concurrency or config_dict.get("num_processes", 10)),
            len(indices),
        ),
    )
    qa_concurrency = max(
        1, int(qa_concurrency or config_dict.get("qa_parallel_nums", 4))
    )
    sample_max_retries = max(0, int(sample_max_retries))
    per_endpoint = int(config_dict.get("per_endpoint_concurrency", 16))
    total_llm_concurrency = per_endpoint * len(base_urls)

    before_state = {
        index: _graph_state(source_root, index, dataset[index]["sample_id"])
        for index in indices
    }
    pending = []
    for index in indices:
        result_path, token_path = _sample_paths(output_dir, index)
        if os.path.isfile(result_path) and os.path.isfile(token_path):
            print(f"[resume] Sample {index} already complete.")
        else:
            pending.append((index, 0))
    print(
        f"{retrieval_mode} re-retrieval: source={source_mode}, "
        f"selected={len(indices)}, completed={len(indices) - len(pending)}, "
        f"pending={len(pending)}, sample_workers={sample_concurrency}, "
        f"qa_workers={qa_concurrency}, graph_top_k=10",
        flush=True,
    )
    print(
        f"Healthy LLM endpoints={base_urls}; effective concurrency cap="
        f"{total_llm_concurrency} ({per_endpoint} per endpoint).",
        flush=True,
    )

    failures = []
    if pending:
        with mp.Manager() as manager:
            llm_runtime = {
                "per_endpoint_concurrency": per_endpoint,
                "semaphores": [
                    manager.BoundedSemaphore(per_endpoint)
                    for _ in base_urls
                ],
                "counter": manager.Value("i", 0),
                "lock": manager.Lock(),
            }
            context = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=sample_concurrency,
                mp_context=context,
            ) as executor:
                future_meta = {}

                def submit(index, attempt):
                    future = executor.submit(
                        _process_sample,
                        index,
                        dataset[index],
                        config_dict,
                        source_root,
                        output_dir,
                        llm_runtime,
                        qa_concurrency,
                        max_qa,
                    )
                    future_meta[future] = (index, attempt)
                    print(
                        f"[runner] Submitted sample {index} "
                        f"attempt {attempt + 1}/{sample_max_retries + 1}.",
                        flush=True,
                    )

                for index, attempt in pending:
                    submit(index, attempt)

                while future_meta:
                    done, _ = concurrent.futures.wait(
                        tuple(future_meta),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        index, attempt = future_meta.pop(future)
                        try:
                            future.result()
                            print(
                                f"[runner] Sample {index} complete.",
                                flush=True,
                            )
                        except Exception as exc:
                            print(
                                f"[runner] Sample {index} failed on attempt "
                                f"{attempt + 1}: {type(exc).__name__}: {exc}",
                                flush=True,
                            )
                            if attempt < sample_max_retries:
                                submit(index, attempt + 1)
                            else:
                                failures.append((index, str(exc)))

    if failures:
        details = "; ".join(
            f"sample {index}: {error}" for index, error in failures
        )
        raise RuntimeError(
            f"{len(failures)} samples exhausted retries: {details}"
        )

    results = []
    token_usages = []
    from .token_tracker import merge_token_usages, save_token_usage

    for index in indices:
        result_path, token_path = _sample_paths(output_dir, index)
        with open(result_path, "r", encoding="utf-8") as handle:
            results.append(json.load(handle))
        with open(token_path, "r", encoding="utf-8") as handle:
            token_usages.append(json.load(handle))

    after_state = {
        index: _graph_state(source_root, index, dataset[index]["sample_id"])
        for index in indices
    }
    if before_state != after_state:
        raise RuntimeError(
            "A persisted source graph changed during read-only re-retrieval"
        )

    _save_json(os.path.join(output_dir, "result.json"), results)
    merged_usage = merge_token_usages(token_usages)
    merged_usage.update(
        {
            "source_mode": source_mode,
            "retrieval_mode": retrieval_mode,
            "retrieval_top_k": 10,
            "sample_count": len(results),
        }
    )
    save_token_usage(
        os.path.join(output_dir, "token_usage.json"), merged_usage
    )
    manifest_name = (
        "graph_plus_original_manifest.json"
        if include_original_text
        else "graph_only_manifest.json"
    )
    _save_json(
        os.path.join(output_dir, manifest_name),
        {
            "source_mode": source_mode,
            "source_root": source_root,
            "retrieval_mode": retrieval_mode,
            "retrieval_top_k": 10,
            "include_original_text": bool(include_original_text),
            "original_text_link": "edge.episode_ids -> mid_term.pages.content",
            "dialogue_retrieval": False,
            "llm_endpoints": base_urls,
            "per_endpoint_concurrency": per_endpoint,
            "effective_llm_concurrency": total_llm_concurrency,
            "selected_sample_indices": indices,
            "source_graphs_unchanged": True,
            "source_graph_state": list(after_state.values()),
            "result_file": os.path.join(output_dir, "result.json"),
        },
    )
    print(
        f"{retrieval_mode} result saved: {output_dir}/result.json "
        f"({len(results)} samples).",
        flush=True,
    )
    return results


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Re-query persisted segment/message graphs with graph-only top-10 "
            "retrieval; no indexing is performed."
        )
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument(
        "--source-mode",
        required=True,
        choices=("segment_graph", "message_graph"),
    )
    parser.add_argument(
        "--source-root",
        help="Root containing segment_graph and message_graph source directories.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--include-original-text",
        action="store_true",
        help="Attach source pages linked by each selected edge's episode_ids.",
    )
    parser.add_argument(
        "--llm-base-url",
        help=(
            "Optional comma-separated endpoint override for this read-only run."
        ),
    )
    parser.add_argument("--sample-concurrency", type=int)
    parser.add_argument("--qa-concurrency", type=int)
    parser.add_argument("--sample-max-retries", type=int, default=2)
    parser.add_argument(
        "--sample-indices",
        help="Comma-separated zero-based sample indices; default is all.",
    )
    parser.add_argument(
        "--max-qa",
        type=int,
        help="Optional QA limit per sample for smoke testing.",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    run_graph_only(
        config_path=args.config_path,
        source_mode=args.source_mode,
        source_root=args.source_root,
        output_dir=args.output_dir,
        include_original_text=args.include_original_text,
        sample_concurrency=args.sample_concurrency,
        qa_concurrency=args.qa_concurrency,
        sample_max_retries=args.sample_max_retries,
        sample_indices=args.sample_indices,
        max_qa=args.max_qa,
        llm_base_url=args.llm_base_url,
    )


if __name__ == "__main__":
    main()
