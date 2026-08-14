import asyncio
import copy
import glob
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

from Method.memgas.llm_client import OpenAICompatibleLLM
from Method.memgas.main import filter_retrieved_context, generate_answer
from simplify import build_client, process_json_file


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULT_DIR = os.path.join(
    ROOT,
    "Result",
    "LONGMEMEVAL",
    "memgas",
    "longmemeval_qwen3.5_9b",
    "top_k_10",
)
RESULT_PATH = os.path.join(RESULT_DIR, "result_simplified.json")
BACKUP_TAG = "before_filter_response_rerun_20260728"
SOURCE_BACKUP = f"{RESULT_PATH}.{BACKUP_TAG}"
FILTER_RESPONSE_STATE = os.path.join(RESULT_DIR, "filter_response_rerun_state_20260728.json")
FILTER_RESPONSE_RAW = os.path.join(
    RESULT_DIR,
    "result_after_filter_response_before_simplify_20260728.json",
)
SIMPLIFY_CHECKPOINT = f"{RESULT_PATH}.resimplifying_20260728"
SIMPLIFY_TOKEN_FILE = os.path.join(RESULT_DIR, "simplify_token_tracker_20260728.json")
FILTER_RESPONSE_MARKER = os.path.join(RESULT_DIR, ".filter_response_rerun_complete_20260728")
SIMPLIFY_MARKER = os.path.join(RESULT_DIR, ".resimplify_complete_20260728")
EVALUATION_MARKER = os.path.join(RESULT_DIR, ".reevaluation_complete_20260728")
PIPELINE_MARKER = os.path.join(RESULT_DIR, ".filter_response_pipeline_complete_20260728")

LLM_BASE_URL = os.getenv("MEMGAS_LLM_BASE_URL")
LLM_MODEL = "Qwen3.5-9B"
LLM_API_KEY = "EMPTY"
EMBEDDING_BASE_URL = os.getenv("MEMGAS_EMBEDDING_BASE_URL")
EMBEDDING_MODEL = "/path/to/local/all-MiniLM-L6-v2"
ANSWER_WORKERS = 12


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path, payload):
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def write_marker(path):
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        handle.write("complete\n")
    os.replace(temporary_path, path)


def backup_once(path):
    backup_path = f"{path}.{BACKUP_TAG}"
    if os.path.exists(path) and not os.path.exists(backup_path):
        shutil.copy2(path, backup_path)
    return backup_path


def retrieved_to_text(retrieved):
    if isinstance(retrieved, list):
        return "".join(
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            for item in retrieved
        )
    if isinstance(retrieved, str):
        return retrieved
    if retrieved:
        return json.dumps(retrieved, ensure_ascii=False)
    return ""


def build_mem():
    if not LLM_BASE_URL:
        raise RuntimeError("MEMGAS_LLM_BASE_URL must name an explicit local endpoint")
    return SimpleNamespace(
        llm=OpenAICompatibleLLM(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            model=LLM_MODEL,
            max_tokens=500,
            temperature=0.0,
            max_retries=5,
            retry_wait_sec=2.0,
            context_window=16384,
            prompt_token_buffer=128,
            use_qwen_thinking_control=True,
        )
    )


def rerun_one(mem, sample_index, qa_index, qa):
    question = qa.get("question", "")
    question_date = qa.get("question_date")
    retrieved_context = retrieved_to_text(qa.get("retrieved"))
    filtered = filter_retrieved_context(
        mem=mem,
        question=question,
        question_date=question_date,
        retrieved_context=retrieved_context,
        tracker=None,
    )
    response = generate_answer(
        mem=mem,
        question=question,
        question_date=question_date,
        filtered_context=filtered,
        hits=[],
        tracker=None,
    )
    return (
        f"{sample_index}:{qa_index}",
        {
            "filtered_retrieved": filtered,
            "response": response,
        },
    )


def load_filter_response_state():
    if os.path.exists(FILTER_RESPONSE_STATE):
        state = read_json(FILTER_RESPONSE_STATE)
        if isinstance(state, dict) and isinstance(state.get("updates"), dict):
            return state
    return {"updates": {}}


def rerun_filter_and_response():
    if os.path.exists(FILTER_RESPONSE_MARKER) and os.path.exists(FILTER_RESPONSE_RAW):
        print(f"Filter/response rerun already complete: {FILTER_RESPONSE_RAW}")
        return

    source = read_json(SOURCE_BACKUP)
    state = load_filter_response_state()
    updates = state["updates"]
    jobs = []
    for sample_index, sample in enumerate(source):
        for qa_index, qa in enumerate(sample.get("qa", [])):
            key = f"{sample_index}:{qa_index}"
            if key not in updates:
                jobs.append((sample_index, qa_index, qa))

    print(
        f"Reusing saved retrieval results: {len(updates)} completed, "
        f"{len(jobs)} filter/response jobs remaining."
    )
    mem = build_mem()
    failures = []
    if jobs:
        with ThreadPoolExecutor(max_workers=ANSWER_WORKERS) as executor:
            future_to_key = {
                executor.submit(rerun_one, mem, sample_index, qa_index, qa): (
                    sample_index,
                    qa_index,
                )
                for sample_index, qa_index, qa in jobs
            }
            completed_now = 0
            for future in as_completed(future_to_key):
                sample_index, qa_index = future_to_key[future]
                key = f"{sample_index}:{qa_index}"
                try:
                    returned_key, update = future.result()
                    updates[returned_key] = update
                    completed_now += 1
                    write_json_atomic(FILTER_RESPONSE_STATE, state)
                    if completed_now % 10 == 0 or completed_now == len(jobs):
                        print(
                            f"Filter/response progress: {len(updates)}/"
                            f"{len(updates) + len(jobs) - completed_now}"
                        )
                except Exception as exc:
                    failures.append((key, repr(exc)))
                    print(f"Filter/response failed for {key}: {exc}")

    if failures:
        raise RuntimeError(f"{len(failures)} filter/response jobs failed: {failures[:5]}")

    improved = copy.deepcopy(source)
    expected = 0
    for sample_index, sample in enumerate(improved):
        for qa_index, qa in enumerate(sample.get("qa", [])):
            expected += 1
            key = f"{sample_index}:{qa_index}"
            update = updates.get(key)
            if not update:
                raise RuntimeError(f"Missing filter/response result for {key}")
            qa.update(update)

    if len(updates) != expected:
        raise RuntimeError(
            f"Filter/response validation failed: {len(updates)} updates for {expected} QA."
        )
    write_json_atomic(FILTER_RESPONSE_RAW, improved)
    write_marker(FILTER_RESPONSE_MARKER)
    print(f"Filter/response rerun complete: {expected} QA.")


async def simplify_improved_results():
    if os.path.exists(SIMPLIFY_MARKER) and os.path.exists(RESULT_PATH):
        print(f"Resimplification already complete: {RESULT_PATH}")
        return

    client = build_client(
        LLM_API_KEY,
        LLM_BASE_URL,
        per_endpoint_concurrency=12,
    )
    try:
        await process_json_file(
            input_file=FILTER_RESPONSE_RAW,
            output_file=SIMPLIFY_CHECKPOINT,
            client=client,
            model_name=LLM_MODEL,
            temperature=0.0,
            max_tokens=150,
            concurrency=12,
            resume=True,
            token_file=SIMPLIFY_TOKEN_FILE,
        )
    finally:
        await client.close()

    original = read_json(FILTER_RESPONSE_RAW)
    simplified = read_json(SIMPLIFY_CHECKPOINT)
    original_qa = sum(len(sample.get("qa", [])) for sample in original)
    simplified_qa = sum(len(sample.get("qa", [])) for sample in simplified)
    if len(original) != len(simplified) or original_qa != simplified_qa:
        raise RuntimeError(
            f"Resimplification validation failed: "
            f"{len(simplified)}/{len(original)} samples, "
            f"{simplified_qa}/{original_qa} QA."
        )

    os.replace(SIMPLIFY_CHECKPOINT, RESULT_PATH)
    write_marker(SIMPLIFY_MARKER)
    print(f"Resimplification complete: {simplified_qa} QA.")


def backup_evaluation_outputs():
    patterns = [
        "memgas_longmemeval_judged.json",
        "memgas_longmemeval_statistics.txt",
        "memgas_longmemeval_statistics.json",
    ]
    for pattern in patterns:
        for path in glob.glob(os.path.join(RESULT_DIR, pattern)):
            backup_once(path)


def rerun_evaluation():
    if os.path.exists(EVALUATION_MARKER):
        print(f"Re-evaluation already complete: {RESULT_DIR}")
        return

    if not EMBEDDING_BASE_URL:
        raise RuntimeError(
            "MEMGAS_EMBEDDING_BASE_URL must name an explicit local endpoint"
        )
    backup_evaluation_outputs()
    env = os.environ.copy()
    env["EVAL_EMBEDDING_BASE_URL"] = EMBEDDING_BASE_URL
    env["EVAL_EMBEDDING_API_KEY"] = "EMPTY"
    env["EVAL_DEVICE"] = "cpu"
    subprocess.run(
        [
            sys.executable,
            "-u",
            "eval.py",
            "--dataset",
            "lme",
            "--method",
            "memgas",
            "--version",
            "longmemeval_qwen3.5_9b/top_k_10",
            "--embedding_model",
            EMBEDDING_MODEL,
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    write_marker(EVALUATION_MARKER)
    print(f"Re-evaluation complete: {RESULT_DIR}")


async def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    if os.path.exists(PIPELINE_MARKER):
        print(f"Pipeline already complete: {RESULT_DIR}")
        return

    backup_once(RESULT_PATH)
    if not os.path.exists(SOURCE_BACKUP):
        raise FileNotFoundError(f"Missing source result backup: {SOURCE_BACKUP}")

    rerun_filter_and_response()
    await simplify_improved_results()
    rerun_evaluation()
    write_marker(PIPELINE_MARKER)
    print(f"Filter/response pipeline complete: {RESULT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
