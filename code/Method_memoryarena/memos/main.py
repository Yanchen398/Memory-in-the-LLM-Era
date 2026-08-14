import asyncio
import json
import os

from .configuration import (
    DEFAULT_RESULT_ROOT,
    build_retrieval_runtime_config,
    build_runtime_config,
    load_runtime_config_file,
)
from .locomo_ingestion import ingestion
from .locomo_responses import response
from .locomo_search import search


def evaluate_result(runtime_config):
    if not runtime_config.get("auto_eval"):
        return

    formatted_path = runtime_config["formatted_results_path"]
    simplified_path = runtime_config["simplified_results_path"]
    if runtime_config.get("simplify_before_eval", True):
        from simplify import build_client, process_json_file

        simplify_client = build_client(
            runtime_config["response_api_key"],
            runtime_config["response_base_url"],
        )
        asyncio.run(
            process_json_file(
                formatted_path,
                simplified_path,
                simplify_client,
                runtime_config["response_model"],
                runtime_config["simplify_temperature"],
                runtime_config["simplify_max_tokens"],
            )
        )
    else:
        raise ValueError(
            "auto_eval requires simplify_before_eval=true; refusing to copy raw "
            "predictions into result_simplified.json"
        )

    from eval import main as eval_main

    version = os.path.relpath(runtime_config["result_dir"], DEFAULT_RESULT_ROOT)
    eval_main(
        "loco",
        "memos",
        version,
        runtime_config["eval_embedding_model"],
        result_path=simplified_path,
    )


def write_experiment_summary(runtime_config, completed_configs):
    summary = {
        "dataset_path": runtime_config["dataset_path"],
        "model": runtime_config["llm_model"],
        "retrieve_top_ks": runtime_config["retrieve_top_ks"],
        "completed_top_ks": list(completed_configs),
        "results": {},
    }

    for top_k, config in completed_configs.items():
        result_dir = config["result_dir"]
        latency_path = os.path.join(result_dir, "retrieval_latency.json")
        statistics_path = os.path.join(result_dir, "memos_locomo_statistics.json")

        entry = {"result_dir": result_dir}
        if os.path.exists(latency_path):
            with open(latency_path, "r", encoding="utf-8") as f:
                entry["retrieval_latency"] = json.load(f)
        if os.path.exists(statistics_path):
            with open(statistics_path, "r", encoding="utf-8") as f:
                statistics = json.load(f)
            entry["metrics"] = statistics.get("overall_statistics", {})
            entry["metrics_by_category"] = statistics.get(
                "overall_category_averages", {}
            )
        summary["results"][str(top_k)] = entry

    summary_path = os.path.join(runtime_config["result_dir"], "experiment_summary.json")
    os.makedirs(runtime_config["result_dir"], exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Updated experiment summary: {summary_path}")


def run_memos(
    version="default",
    num_workers=4,
    top_k=10,
    dataset_path=None,
    config=None,
    config_path=None,
):
    if config is None and config_path:
        config = load_runtime_config_file(config_path)

    runtime_config = build_runtime_config(
        config
        or {
            "version": version,
            "num_workers": num_workers,
            "top_k": top_k,
            "dataset_path": dataset_path,
        },
        config_path=config_path,
    )

    if runtime_config.get("skip_ingestion"):
        print("Skipping Memos ingestion and reusing the existing index...")
    else:
        print("Building Memos index once for all retrieval top-k settings...")
        ingestion(runtime_config)

    results = {}
    completed_configs = {}
    for top_k_value in runtime_config["retrieve_top_ks"]:
        retrieval_config = build_retrieval_runtime_config(runtime_config, top_k_value)
        print(f"Running Memos retrieval and response generation for top_k={top_k_value}...")
        search(retrieval_config)
        asyncio.run(response(retrieval_config))
        evaluate_result(retrieval_config)
        results[top_k_value] = retrieval_config["result_dir"]
        completed_configs[top_k_value] = retrieval_config
        write_experiment_summary(runtime_config, completed_configs)

    print(f"Completed top-k settings: {runtime_config['retrieve_top_ks']}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version identifier for saving results (e.g., 1010)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers to process users"
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of results to retrieve in search queries"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the LOCOMO dataset JSON file",
    )
    parser.add_argument("--config_path", type=str, help="Path to the memos config file")
    args = parser.parse_args()

    run_memos(
        version=args.version,
        num_workers=args.workers,
        top_k=args.top_k,
        dataset_path=args.dataset_path,
        config_path=args.config_path,
    )
