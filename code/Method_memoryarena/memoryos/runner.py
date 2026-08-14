import argparse
import concurrent.futures
import difflib
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path

import yaml

DATASET_NAMES = (
    "bundled_shopping",
    "progressive_search",
    "group_travel_planner",
    "formal_reasoning_math",
    "formal_reasoning_phys",
)
TRAVEL_SLOTS = (
    "breakfast", "lunch", "dinner", "accommodation", "transportation", "attraction"
)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Ignoring malformed line {line_number} in {path}: {exc}", flush=True)
    return values


def _extract_json(text):
    text = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _load_tasks(config):
    root = Path(config["dataset_root"])
    requested = config.get("datasets") or list(DATASET_NAMES)
    unknown = sorted(set(requested) - set(DATASET_NAMES))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    limit = config.get("max_tasks_per_dataset")
    tasks = []
    for dataset_name in requested:
        rows = _read_jsonl(root / dataset_name / "data.jsonl")
        if limit is not None:
            rows = rows[:max(0, int(limit))]
        for row_index, sample in enumerate(rows):
            tasks.append({
                "task_id": f"{dataset_name}:{sample.get('id', row_index)}",
                "dataset_name": dataset_name,
                "row_index": row_index,
                "sample": sample,
            })
    return tasks


def _task_query(dataset_name, sample, index):
    question = sample["questions"][index]
    if dataset_name == "bundled_shopping":
        instruction = (
            "Solve this bundled-shopping subtask using relevant prior purchases from memory. "
            "Evaluate every listed option and all compatibility, budget, price, and rating constraints. "
            "Return JSON only with selected_product, target_asin (only if known), and concise_reasoning. "
            "Never invent an ASIN."
        )
    elif dataset_name == "progressive_search":
        instruction = (
            "Solve this progressive information-seeking subtask. Reuse candidate identities and evidence "
            "from earlier sessions and apply every current condition. Return the final entity or entities "
            "first, followed by a concise evidence summary."
        )
    elif dataset_name == "group_travel_planner":
        instruction = (
            "Plan this new group member's itinerary. Preserve the base itinerary except where this traveler "
            "changes or joins an earlier member's activity. Reuse exact names and values from memory. "
            "Return JSON only as a list of day objects with keys days, current_city, transportation, "
            "breakfast, attraction, lunch, dinner, accommodation."
        )
    else:
        backgrounds = sample.get("backgrounds") or [""] * len(sample["questions"])
        question = f"QUESTION:\n{question}\n\nBACKGROUND AND DEFINITIONS:\n{backgrounds[index]}"
        instruction = (
            "Solve this sequential formal-reasoning subtask rigorously. Reuse valid intermediate results "
            "from earlier sessions. Show the essential derivation and finish with a clearly marked final answer."
        )
    return f"MEMORYARENA TASK INSTRUCTIONS:\n{instruction}\n\nCURRENT SUBTASK:\n{question}"


def _collect_retrieved(result):
    if not result:
        return []
    values = []
    for page in result.get("retrieved_pages", []):
        values.append({key: page.get(key) for key in ("user_input", "agent_response", "timestamp")})
    for key in ("retrieved_user_knowledge", "retrieved_assistant_knowledge"):
        for item in result.get(key, []):
            values.append({key: item.get("knowledge")})
    return values


def _format_memory_context(memo, retrieval_result):
    sections = []
    short_term = memo.short_term_memory.get_all()
    if short_term:
        sections.append("SHORT-TERM SESSION HISTORY:\n" + "\n\n".join(
            f"Input: {item.get('user_input', '')}\nOutput: {item.get('agent_response', '')}"
            for item in short_term
        ))
    retrieved = _collect_retrieved(retrieval_result)
    if retrieved:
        sections.append("TOP-K RETRIEVED LONG-TERM MEMORY:\n" + "\n\n".join(
            json.dumps(item, ensure_ascii=False) for item in retrieved
        ))
    if not sections:
        return "No prior memory is available for the first subtask."
    return "\n\n".join(sections)


def _generate_task(task, config):
    from memoryos import Memoryos
    from .main_lme_runner import _configure_library_runtime

    _configure_library_runtime(
        config["embedding_model_name"],
        config.get("embedding_api_key", "EMPTY"),
        config.get("embedding_base_url", "http://127.0.0.1:7999/v1"),
        config["llm_model"],
    )
    dataset_name = task["dataset_name"]
    sample = task["sample"]
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", task["task_id"])
    memory_path = Path(config["memory_path"]) / safe_id
    shutil.rmtree(memory_path, ignore_errors=True)
    memory_path.mkdir(parents=True, exist_ok=True)
    memo = Memoryos(
        user_id=f"memoryarena_{safe_id}",
        assistant_id=f"memoryarena_assistant_{safe_id}",
        openai_api_key=config.get("llm_api_key", "EMPTY"),
        openai_base_url=config["llm_base_url"],
        data_storage_path=str(memory_path),
        llm_model=config["llm_model"],
        short_term_capacity=int(config.get("short_term_capacity", 7)),
        mid_term_heat_threshold=float(config.get("mid_term_heat_threshold", 5)),
        retrieval_queue_capacity=int(config.get("retrieval_top_k", 10)),
    )
    if dataset_name == "group_travel_planner":
        base = sample["base_person"]
        memo.add_memory(
            user_input=f"Base traveler {base['name']}: {base['query']}",
            agent_response=json.dumps(base["daily_plans"], ensure_ascii=False),
            meta_data={"kind": "base_traveler", "name": base["name"]},
        )
    subtasks = []
    for index, question in enumerate(sample["questions"]):
        query_text = _task_query(dataset_name, sample, index)
        retrieval_started = time.perf_counter()
        retrieval_result = memo.retriever.retrieve_context(
            user_query=query_text, user_id=memo.user_id,
        )
        retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
        memory_context = _format_memory_context(memo, retrieval_result)
        started = time.perf_counter()
        response = memo.client.chat_completion(
            model=config["llm_model"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the task agent in the MemoryArena memory-agent-environment loop. "
                        "Solve the current subtask yourself using the supplied memory context. "
                        "Follow the requested output format completely: never announce a JSON/list/derivation "
                        "without actually emitting it, and never ask the user for missing information."
                    ),
                },
                {
                    "role": "user",
                    "content": f"{query_text}\n\nMEMORY CONTEXT:\n{memory_context}",
                },
            ],
            temperature=float(config.get("temperature", 0.0)),
            max_tokens=int(config.get("max_response_tokens", 3000)),
        )
        if str(response).startswith("Error: Could not get response from LLM"):
            raise RuntimeError(f"LLM failure at {task['task_id']} subtask {index}")
        memo.add_memory(
            user_input=query_text,
            agent_response=response,
            meta_data={
                "benchmark": "MemoryArena",
                "dataset": dataset_name,
                "task_id": task["task_id"],
                "subtask_index": index,
            },
        )
        subtasks.append({
            "subtask_index": index,
            "question": question,
            "response": response,
            "retrieved": _collect_retrieved(retrieval_result),
            "retrieval_top_k": int(config.get("retrieval_top_k", 10)),
            "retrieval_latency_ms": retrieval_latency_ms,
            "end_to_end_latency_ms": (time.perf_counter() - started) * 1000,
        })
        print(f"[memoryarena] {task['task_id']} subtask {index + 1}/{len(sample['questions'])} complete", flush=True)
    return {
        "task_id": task["task_id"], "dataset_name": dataset_name,
        "source_id": sample.get("id"), "row_index": task["row_index"],
        "subtasks": subtasks,
    }


def _generation_entry(result_queue, task, config):
    try:
        result_queue.put({"status": "ok", "result": _generate_task(task, config)})
    except BaseException as exc:
        result_queue.put({
            "status": "error", "task_id": task["task_id"],
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
        })


def _result_path(config, dataset_name):
    return Path(config["output_path"]) / dataset_name / "generated_tasks.jsonl"


def _load_generated(config):
    generated = {}
    for dataset_name in config.get("datasets") or DATASET_NAMES:
        for value in _read_jsonl(_result_path(config, dataset_name)):
            if value.get("task_id"):
                generated[value["task_id"]] = value
    return generated


def run_generation(config):
    tasks = _load_tasks(config)
    generated = _load_generated(config)
    pending = deque((task, 0) for task in tasks if task["task_id"] not in generated)
    concurrency = max(1, int(config.get("sample_concurrency", 16)))
    max_retries = max(0, int(config.get("sample_max_retries", 2)))
    print(f"[memoryarena] Generation: {len(generated)} completed, {len(pending)} pending, {concurrency} workers", flush=True)
    context = mp.get_context("spawn")
    active, failures = {}, []

    def launch(task, attempt):
        result_queue = context.Queue(maxsize=1)
        process = context.Process(target=_generation_entry, args=(result_queue, task, config))
        process.start()
        active[task["task_id"]] = {"process": process, "queue": result_queue, "task": task, "attempt": attempt}
        print(f"[memoryarena] Launched {task['task_id']} attempt {attempt + 1}/{max_retries + 1} pid={process.pid}", flush=True)

    try:
        while pending or active:
            while pending and len(active) < concurrency:
                launch(*pending.popleft())
            finished = []
            for task_id, job in list(active.items()):
                message = None
                try:
                    message = job["queue"].get_nowait()
                except queue.Empty:
                    pass
                process = job["process"]
                if message is not None:
                    process.join(timeout=10)
                    if process.is_alive():
                        process.terminate(); process.join(timeout=5)
                    finished.append((task_id, message))
                elif not process.is_alive():
                    process.join()
                    try:
                        message = job["queue"].get(timeout=1)
                    except queue.Empty:
                        message = {"status": "error", "error": f"worker exited without result ({process.exitcode})"}
                    finished.append((task_id, message))
            if not finished:
                time.sleep(1); continue
            for task_id, message in finished:
                job = active.pop(task_id); job["queue"].close()
                if message["status"] == "ok":
                    value = message["result"]
                    generated[task_id] = value
                    _append_jsonl(_result_path(config, value["dataset_name"]), value)
                    print(f"[memoryarena] Persisted {task_id}; {len(generated)}/{len(tasks)}", flush=True)
                else:
                    print(f"[memoryarena] {task_id} failed: {message.get('error')}\n{message.get('traceback', '')}", flush=True)
                    if job["attempt"] < max_retries:
                        pending.appendleft((job["task"], job["attempt"] + 1))
                    else:
                        failures.append((task_id, message.get("error")))
    finally:
        for job in active.values():
            if job["process"].is_alive(): job["process"].terminate()
            job["process"].join(timeout=5); job["queue"].close()
    if failures:
        raise RuntimeError(f"Generation failures exhausted retries: {failures}")
    return tasks, generated


def _similarity(left, right):
    if left is None or right is None: return 0.0
    left, right = str(left).strip().lower(), str(right).strip().lower()
    if left == right == "-": return 1.0
    length = min(len(left), len(right))
    return difflib.SequenceMatcher(None, left[:length], right[:length]).ratio() if length else 0.0


def _get_day(plan, day_index):
    if not isinstance(plan, list): return None
    return next((day for day in plan if isinstance(day, dict) and (day.get("day") == day_index or day.get("days") == day_index)), None)


def _extract_plan(response):
    value = _extract_json(response)
    if isinstance(value, list): return value
    if isinstance(value, dict):
        for key in ("daily_plans", "plan", "itinerary"):
            if isinstance(value.get(key), list): return value[key]
    return None


def _travel_scores(base_plan, gold_plan, response):
    predicted = _extract_plan(response)
    full_pass = bool(gold_plan and predicted)
    passed_constraints = total_constraints = 0
    for gold_day in gold_plan or []:
        day_index = gold_day.get("days") or gold_day.get("day")
        base_day, predicted_day = _get_day(base_plan, day_index), _get_day(predicted, day_index)
        if predicted_day is None: full_pass = False
        for slot in TRAVEL_SLOTS:
            gold_value = gold_day.get(slot, "-")
            predicted_value = predicted_day.get(slot, "-") if predicted_day else None
            passed = _similarity(gold_value, predicted_value) >= 0.7
            full_pass = full_pass and passed
            base_value = base_day.get(slot, "-") if base_day else None
            if _similarity(base_value, gold_value) < 0.7:
                total_constraints += 1; passed_constraints += int(passed)
    soft_score = passed_constraints / total_constraints if total_constraints else 1.0
    return bool(full_pass), soft_score, predicted


def _judge_prompt(dataset_name, question, gold, response):
    if dataset_name == "bundled_shopping":
        criterion = "The gold identifies the unique product by ASIN and attributes. Correct requires the same selected product."
    elif dataset_name == "progressive_search":
        criterion = "Correct requires the same final entity or entity set; ignore harmless wording."
    else:
        criterion = "Correct requires a mathematically equivalent final result; ignore notation but not substantive errors."
    return (
        "You are a strict MemoryArena evaluator. " + criterion + " Return JSON only: "
        "{\"correct\":true or false,\"score\":0 to 1,\"reason\":\"brief\"}.\n\n"
        f"QUESTION:\n{question}\n\nGOLD:\n{json.dumps(gold, ensure_ascii=False)}\n\nPREDICTION:\n{response}"
    )


def _judge_one(client, model, dataset_name, question, gold, response):
    if dataset_name == "bundled_shopping" and isinstance(gold, dict):
        asin = str(gold.get("target_asin", "")).upper()
        if asin and asin in str(response).upper():
            return {"correct": True, "score": 1.0, "reason": "Exact target ASIN"}
    last_error = None
    for attempt in range(3):
        try:
            text = client.chat_completion(
                model=model, messages=[{"role": "user", "content": _judge_prompt(dataset_name, question, gold, response)}],
                temperature=0, max_tokens=256,
            )
            value = _extract_json(text)
            if not isinstance(value, dict) or not isinstance(value.get("correct"), bool):
                raise ValueError(f"Invalid judge JSON: {text[:300]}")
            value["score"] = min(1.0, max(0.0, float(value.get("score", int(value["correct"])))))
            value["raw"] = text
            return value
        except Exception as exc:
            last_error = exc; time.sleep(1 + attempt)
    raise RuntimeError(f"Judge failed: {last_error}")


def _evaluate_task(task, generated, config):
    from .utils import OpenAIClient
    dataset_name, sample = task["dataset_name"], task["sample"]
    generated_by_index = {item["subtask_index"]: item for item in generated["subtasks"]}
    judged = []
    if dataset_name == "group_travel_planner":
        base_plan = sample["base_person"]["daily_plans"]
        for index, gold in enumerate(sample["answers"]):
            item = generated_by_index[index]
            correct, soft_score, plan = _travel_scores(base_plan, gold, item["response"])
            judged.append({**item, "gold_answer": gold, "is_correct": correct, "soft_score": soft_score, "parsed_plan": plan})
    else:
        client = OpenAIClient(config.get("llm_api_key", "EMPTY"), config["llm_base_url"])
        for index, gold in enumerate(sample["answers"]):
            item = generated_by_index[index]
            decision = _judge_one(client, config.get("judge_model", config["llm_model"]), dataset_name, item["question"], gold, item["response"])
            judged.append({**item, "gold_answer": gold, "is_correct": decision["correct"], "judge_score": decision["score"], "judge_reason": decision.get("reason"), "judge_raw": decision.get("raw")})
    return {"task_id": task["task_id"], "dataset_name": dataset_name, "source_id": sample.get("id"), "row_index": task["row_index"], "subtasks": judged}


def _evaluation_path(config, task):
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", task["task_id"])
    return Path(config["output_path"]) / task["dataset_name"] / "evaluated_tasks" / f"{safe_id}.json"


def run_evaluation(tasks, generated, config):
    evaluated, pending = {}, []
    for task in tasks:
        path = _evaluation_path(config, task)
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if len(value.get("subtasks", [])) == len(task["sample"]["questions"]):
                    evaluated[task["task_id"]] = value; continue
            except Exception: pass
        pending.append(task)
    concurrency = max(1, int(config.get("judge_concurrency", 16)))
    print(f"[memoryarena-eval] {len(evaluated)} completed, {len(pending)} pending, {concurrency} workers", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_evaluate_task, task, generated[task["task_id"]], config): task for task in pending}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]; value = future.result()
            evaluated[task["task_id"]] = value
            _atomic_json(_evaluation_path(config, task), value)
            print(f"[memoryarena-eval] Persisted {task['task_id']}; {len(evaluated)}/{len(tasks)}", flush=True)
    return evaluated


def _aggregate(tasks, evaluated, config):
    grouped = defaultdict(list)
    for task in tasks: grouped[task["dataset_name"]].append(evaluated[task["task_id"]])
    metrics = {}
    for dataset_name, task_results in grouped.items():
        progress, success, soft, at_k, retrieval, end_to_end = [], [], [], defaultdict(list), [], []
        for task_result in task_results:
            subtasks = task_result["subtasks"]
            correct = [bool(item.get("is_correct")) for item in subtasks]
            progress.append(sum(correct) / len(correct) if correct else 0.0)
            success.append(float(correct[-1]) if dataset_name in {"progressive_search", "formal_reasoning_math", "formal_reasoning_phys"} and correct else float(bool(correct) and all(correct)))
            if dataset_name == "group_travel_planner":
                soft.append(sum(float(item.get("soft_score", 0)) for item in subtasks) / len(subtasks) if subtasks else 0.0)
            for item in subtasks:
                at_k[int(item["subtask_index"]) + 1].append(float(bool(item.get("is_correct"))))
                retrieval.append(float(item.get("retrieval_latency_ms", 0))); end_to_end.append(float(item.get("end_to_end_latency_ms", 0)))
        value = {
            "task_count": len(task_results), "subtask_count": sum(len(x["subtasks"]) for x in task_results),
            "progress_score": sum(progress) / len(progress), "success_rate": sum(success) / len(success),
            "success_rate_at_k": {str(k): sum(v) / len(v) for k, v in sorted(at_k.items())},
            "average_retrieval_latency_ms": sum(retrieval) / len(retrieval),
            "average_end_to_end_latency_ms": sum(end_to_end) / len(end_to_end),
        }
        if soft: value["soft_progress_score"] = sum(soft) / len(soft)
        metrics[dataset_name] = value
    dataset_values = list(metrics.values())
    metrics["macro_average"] = {
        "progress_score": sum(x["progress_score"] for x in dataset_values) / len(dataset_values),
        "success_rate": sum(x["success_rate"] for x in dataset_values) / len(dataset_values),
    }
    _atomic_json(Path(config["output_path"]) / "metrics.json", metrics)
    return metrics


def run(config):
    Path(config["output_path"]).mkdir(parents=True, exist_ok=True)
    Path(config["memory_path"]).mkdir(parents=True, exist_ok=True)
    _atomic_json(Path(config["output_path"]) / "run_metadata.json", {
        "protocol": "dataset_native_sequential_sessions",
        "memory_backend": "memoryos_library",
        "memory_import": "from memoryos import Memoryos",
        "retrieval_top_k": int(config.get("retrieval_top_k", 10)),
        "llm_model": config["llm_model"],
        "llm_base_url": config["llm_base_url"],
        "judge_model": config.get("judge_model", config["llm_model"]),
        "judge_feedback_written_to_memory": False,
        "official_environment_assets_available": False,
        "limitation": (
            "WebShop, web-search, and travel external tool assets are absent; this run uses the provided JSONL "
            "as ordered sessions. Scores are not directly comparable to official environment-backed results."
        ),
    })
    tasks, generated = run_generation(config)
    if config.get("skip_evaluation", False): return {"generated_tasks": len(generated)}
    metrics = _aggregate(tasks, run_evaluation(tasks, generated, config), config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run MemoryOS on MemoryArena")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle: config = yaml.safe_load(handle)
    run(config)


if __name__ == "__main__": main()
