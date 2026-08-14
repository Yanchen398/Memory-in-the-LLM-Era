import asyncio
import glob
import json
import os
import shutil

from simplify import build_client, process_json_file


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
BACKUP_TAG = "before_resimplify_20260701"
MODEL_NAME = os.getenv("MEMGAS_SIMPLIFY_MODEL", "Qwen3.5-9B")
BASE_URL = os.getenv("MEMGAS_SIMPLIFY_BASE_URL")
EMBEDDING_MODEL = "/path/to/local/Qwen3-Embedding-0.6B"

TARGETS = [
    ("lme", "longmemeval_qwen3.5_9b/top_k_10"),
    ("loco", "locomo_qwen3.5_27b/top_k_10"),
    ("loco", "locomo_qwen3.5_9b/top_k_1"),
    ("loco", "locomo_qwen3.5_9b/top_k_3"),
    ("loco", "locomo_qwen3.5_9b/top_k_5"),
    ("loco", "locomo_qwen3.5_9b/top_k_10"),
    ("loco", "locomo_qwen3.5_9b/top_k_15"),
]


def result_path(dataset, version):
    dataset_dir = "LONGMEMEVAL" if dataset == "lme" else "LOCOMO"
    return os.path.join(ROOT, "Result", dataset_dir, "memgas", version, "result_simplified.json")


def backup_once(path, suffix):
    backup_path = f"{path}.{suffix}"
    if os.path.exists(path) and not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
    return backup_path


async def simplify_targets():
    if not BASE_URL:
        raise RuntimeError(
            "MEMGAS_SIMPLIFY_BASE_URL must name an explicit local endpoint"
        )
    client = build_client("EMPTY", BASE_URL)
    try:
        for dataset, version in TARGETS:
            source_path = result_path(dataset, version)
            backup_path = backup_once(source_path, BACKUP_TAG)
            checkpoint_path = f"{source_path}.simplifying"
            marker_path = f"{source_path}.simplify_complete_20260701"
            if os.path.exists(marker_path):
                print(f"Skip completed simplification: {source_path}")
                continue
            print(f"Simplifying: {source_path}")
            await process_json_file(
                input_file=backup_path,
                output_file=checkpoint_path,
                client=client,
                model_name=MODEL_NAME,
                temperature=0.0,
                max_tokens=150,
                concurrency=32,
                resume=True,
            )
            with open(checkpoint_path, "r", encoding="utf-8") as file_obj:
                simplified = json.load(file_obj)
            with open(backup_path, "r", encoding="utf-8") as file_obj:
                original = json.load(file_obj)
            expected_qas = sum(len(sample.get("qa", [])) for sample in original)
            actual_qas = sum(len(sample.get("qa", [])) for sample in simplified)
            if len(simplified) != len(original) or actual_qas != expected_qas:
                raise RuntimeError(f"Simplification validation failed for {source_path}")
            os.replace(checkpoint_path, source_path)
            with open(marker_path, "w", encoding="utf-8") as file_obj:
                file_obj.write("complete\n")
            print(f"Simplification complete: {source_path} ({actual_qas} QA)")
    finally:
        await client.close()


def backup_evaluation_outputs(dataset, version):
    result_dir = os.path.dirname(result_path(dataset, version))
    patterns = ["memgas_*_judged.json", "memgas_*_statistics.txt", "memgas_*_statistics.json"]
    for pattern in patterns:
        for path in glob.glob(os.path.join(result_dir, pattern)):
            backup_once(path, BACKUP_TAG)


def evaluate_targets():
    from eval import main as eval_main

    for dataset, version in TARGETS:
        backup_evaluation_outputs(dataset, version)
        print(f"Re-evaluating dataset={dataset}, version={version}")
        eval_main(dataset, "memgas", version, EMBEDDING_MODEL)


async def main():
    await simplify_targets()
    evaluate_targets()


if __name__ == "__main__":
    asyncio.run(main())
