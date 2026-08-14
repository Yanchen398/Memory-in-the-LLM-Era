import asyncio
import json
import os
import shutil
import subprocess
import sys

import yaml

from simplify import build_client, process_json_file


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(ROOT, "Config", "memgas_locomo_deepseek_v4_flash.yaml")
VERSION = "locomo_deepseek_v4_flash/top_k_10"
RESULT_DIR = os.path.join(ROOT, "Result", "LOCOMO", "memgas", VERSION)
RESULT_PATH = os.path.join(RESULT_DIR, "result_simplified.json")
RAW_BACKUP_PATH = f"{RESULT_PATH}.before_simplify"
SIMPLIFY_CHECKPOINT_PATH = f"{RESULT_PATH}.simplifying"
EXPERIMENT_MARKER = os.path.join(RESULT_DIR, ".experiment_complete")
SIMPLIFY_MARKER = os.path.join(RESULT_DIR, ".simplify_complete")
EVALUATION_MARKER = os.path.join(RESULT_DIR, ".evaluation_complete")
PIPELINE_MARKER = os.path.join(RESULT_DIR, ".pipeline_complete")
SIMPLIFY_TOKEN_FILE = os.path.join(RESULT_DIR, "simplify_token_tracker.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_marker(path):
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        handle.write("complete\n")
    os.replace(temporary_path, path)


def load_result(path):
    with open(path, "r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, list) or not result:
        raise RuntimeError(f"Invalid or empty result file: {path}")
    return result


def qa_count(result):
    return sum(len(sample.get("qa", [])) for sample in result)


def run_experiment():
    if os.path.exists(EXPERIMENT_MARKER) and os.path.exists(RESULT_PATH):
        print(f"Experiment already complete: {RESULT_PATH}")
        return

    print("Starting MemGAS experiment with sample concurrency 5 and retrieval top-k 10.")
    subprocess.run(
        [
            sys.executable,
            "-u",
            "run.py",
            "memgas",
            "--config_file",
            CONFIG_PATH,
        ],
        cwd=ROOT,
        check=True,
    )
    result = load_result(RESULT_PATH)
    print(f"Experiment complete: {len(result)} samples, {qa_count(result)} QA.")
    write_marker(EXPERIMENT_MARKER)


async def simplify_result(config):
    if os.path.exists(SIMPLIFY_MARKER) and os.path.exists(RESULT_PATH):
        print(f"Simplification already complete: {RESULT_PATH}")
        return

    if not os.path.exists(RAW_BACKUP_PATH):
        shutil.copy2(RESULT_PATH, RAW_BACKUP_PATH)

    client = build_client(
        config["llm_api_key"],
        config["llm_base_url"],
        per_endpoint_concurrency=5,
    )
    try:
        await process_json_file(
            input_file=RAW_BACKUP_PATH,
            output_file=SIMPLIFY_CHECKPOINT_PATH,
            client=client,
            model_name=config["llm_model"],
            temperature=0.0,
            max_tokens=150,
            concurrency=5,
            resume=True,
            token_file=SIMPLIFY_TOKEN_FILE,
        )
    finally:
        await client.close()

    original = load_result(RAW_BACKUP_PATH)
    simplified = load_result(SIMPLIFY_CHECKPOINT_PATH)
    if len(simplified) != len(original) or qa_count(simplified) != qa_count(original):
        raise RuntimeError("Simplification validation failed.")

    os.replace(SIMPLIFY_CHECKPOINT_PATH, RESULT_PATH)
    write_marker(SIMPLIFY_MARKER)
    print(f"Simplification complete: {len(simplified)} samples, {qa_count(simplified)} QA.")


def run_evaluation(config):
    if os.path.exists(EVALUATION_MARKER):
        print(f"Evaluation already complete: {RESULT_DIR}")
        return

    subprocess.run(
        [
            sys.executable,
            "-u",
            "eval.py",
            "--dataset",
            "loco",
            "--method",
            "memgas",
            "--version",
            VERSION,
            "--embedding_model",
            config["eval_embedding_model"],
        ],
        cwd=ROOT,
        check=True,
    )
    write_marker(EVALUATION_MARKER)
    print(f"Evaluation complete: {RESULT_DIR}")


async def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    if os.path.exists(PIPELINE_MARKER):
        print(f"Pipeline already complete: {RESULT_DIR}")
        return

    config = load_config()
    if config.get("llm_use_qwen_thinking_control", True):
        raise RuntimeError("DeepSeek config must not enable Qwen thinking control.")
    if int(config.get("num_workers", 0)) != 5:
        raise RuntimeError("DeepSeek config must use exactly 5 sample workers.")
    if str(config.get("retrieve_top_ks")) != "10":
        raise RuntimeError("DeepSeek config must use retrieval top-k 10.")

    run_experiment()
    await simplify_result(config)
    run_evaluation(config)
    write_marker(PIPELINE_MARKER)
    print(f"Pipeline complete: {RESULT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
