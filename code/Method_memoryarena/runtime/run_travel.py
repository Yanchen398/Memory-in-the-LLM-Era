"""
Travel Planner - Memory-Augmented Agent Benchmark
Follows the same pattern as example_math.py:
    env_client.reset → memory.wrap → agent.act → env_client.step → agent.build_memory_entry → memory.add

Requires:
    - Environment server running: python env/env_server.py
    - Memory server running:      python memory/server.py  (if using a memory system)
"""

import os
import re
import json
import time
import uuid
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

_CODE_ROOT = Path(__file__).resolve().parents[2]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from agent.travel_planner import TravelPlannerAgent
from env.env_client import EnvironmentClient
from env.env_systems.travel_planner_env.data_loader import load_travel_data
from env.env_systems.travel_planner_env.combination import combine
from env.env_systems.travel_planner_env.eval import evaluate


def get_memory_system(
    memory_system_name: str,
    user_id: str,
    server_url: str = "http://0.0.0.0:8000",
    timeout: int = 300,
    run_id: str = None,
    config_digest: str = None,
):
    if memory_system_name == "none":
        return None
    from Method_memoryarena.client import MemoryClient
    return MemoryClient(
        user_id=user_id,
        memory_system_name=memory_system_name,
        base_url=server_url,
        timeout=timeout,
        run_id=run_id,
        config_digest=config_digest,
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------



def parse_all_plans(result_text, queries):
    all_results = []
    pattern = r'===\s*([^=]+?)\'s Plan\s*==='
    parts = re.split(pattern, result_text)

    name_plan_pairs = {}
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            name = parts[i].strip()
            plan = parts[i + 1].strip()
            name_plan_pairs[name] = plan

    for idx, round_item in enumerate(queries, start=1):
        name = round_item.get('name', f'Person{idx}')
        query = round_item['query']
        result = name_plan_pairs.get(name, "")
        all_results.append({
            'person_idx': idx,
            'name': name,
            'query': query,
            'result': result,
        })

    return all_results


def write_json_atomic(path, payload):
    """Write a JSON artifact without exposing a partial destination file."""
    path = os.fspath(path)
    output_dir = os.path.dirname(path) or "."
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = os.path.join(
        output_dir,
        f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp",
    )
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4, ensure_ascii=False, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _bounded_error_message(error, limit=8000):
    message = f"{type(error).__name__}: {error}"
    if len(message) <= limit:
        return message
    half = max(1, (limit - 80) // 2)
    return (
        message[:half]
        + "\n...[terminal error evidence truncated]...\n"
        + message[-half:]
    )


def _classify_failure(message):
    lowered = message.lower()
    if "context length" in lowered or "maximum input length" in lowered:
        return "context_limit"
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if "cuda" in lowered:
        return "cuda"
    if "http" in lowered or "server error" in lowered:
        return "http"
    return "agent_error"


def _validate_travel_success_payload(payload, model_name, expected_queries):
    if payload.get("status", "completed") != "completed":
        raise ValueError("Travel result is not completed")
    plans_key = f"{model_name}_sole-planning_results"
    plans = payload.get(plans_key)
    all_results = payload.get("all_results")
    scratchpads = payload.get("scratchpads")
    if expected_queries <= 0:
        expected_queries = max(
            len(plans) if isinstance(plans, list) else 0,
            len(all_results) if isinstance(all_results, list) else 0,
            len(scratchpads) if isinstance(scratchpads, list) else 0,
        )
    if expected_queries <= 0:
        raise ValueError("Travel result contains no traveler plans")
    if not isinstance(plans, list) or len(plans) != expected_queries:
        raise ValueError(
            f"Expected {expected_queries} final plans, got "
            f"{len(plans) if isinstance(plans, list) else 'invalid'}"
        )
    empty_names = [
        str(plan.get("name", plan.get("person_idx", "unknown")))
        for plan in plans
        if not isinstance(plan.get("result"), str) or not plan["result"].strip()
    ]
    if empty_names:
        raise ValueError(f"Empty parsed travel plan(s): {', '.join(empty_names)}")
    if not isinstance(all_results, list) or len(all_results) != expected_queries:
        raise ValueError("Travel all_results does not cover every round")
    if not isinstance(scratchpads, list) or len(scratchpads) != expected_queries:
        raise ValueError("Travel scratchpads does not cover every round")
    scratchpad_errors = [
        str(item.get("error_message"))
        for item in scratchpads
        if isinstance(item, dict) and item.get("error_message")
    ]
    if scratchpad_errors:
        raise ValueError(
            "Travel agent reported terminal error(s): " + "; ".join(scratchpad_errors)
        )
    person_indices = [plan.get("person_idx") for plan in plans]
    if person_indices != list(range(1, expected_queries + 1)):
        raise ValueError(f"Unexpected travel person_idx ordering: {person_indices}")


def _load_existing_travel_result(path, model_name, expected_queries):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") == "failed":
        return payload
    try:
        _validate_travel_success_payload(payload, model_name, expected_queries)
    except (TypeError, ValueError):
        return None
    return payload


def _build_travel_failure_payload(
    args,
    data_idx,
    error,
    stage,
    all_results,
    all_scratchpads,
):
    message = _bounded_error_message(error)
    kind = _classify_failure(message)
    return {
        "metadata": {
            "model_name": args.model_name,
            "judgement_mode": args.judgement_mode,
            "memory_system": args.memory_system,
            "mode": "agent",
            "max_steps": args.max_steps,
            "terminal_failure": True,
            "failure_kind": kind,
            "failure_message": message,
            "failure_policy": "record_as_incorrect_and_continue",
        },
        "status": "failed",
        "data_id": data_idx,
        f"{args.model_name}_sole-planning_results": [],
        "all_results": all_results,
        "scratchpads": all_scratchpads,
        "failure": {"kind": kind, "stage": stage, "message": message},
    }


def format_person_plan(name, daily_plans):
    lines = [f"=== {name}'s Plan ==="]
    for day in daily_plans:
        day_idx = day.get('days') or day.get('day')
        lines.append(f"Day {day_idx}:")
        lines.append(f"Current City: {day.get('current_city', '-')}")
        lines.append(f"Transportation: {day.get('transportation', '-')}")
        lines.append(f"Breakfast: {day.get('breakfast', '-')}")
        lines.append(f"Attraction: {day.get('attraction', '-')}")
        lines.append(f"Lunch: {day.get('lunch', '-')}")
        lines.append(f"Dinner: {day.get('dinner', '-')}")
        lines.append(f"Accommodation: {day.get('accommodation', '-')}")
        lines.append("")
    return "\n".join(lines)


def load_config(config_path):
    """Load a JSON config and return a flat namespace matching CLI args."""
    with open(config_path, 'r') as f:
        cfg = json.load(f)

    agent = cfg.get("agent", {})
    memory = cfg.get("memory", {})
    env = cfg.get("env", {})
    task_specific = cfg.get("task_specific", {})
    output = cfg.get("output", {})
    benchmark_contract = cfg.get("benchmark_contract", {})
    from Method_memoryarena.client import configuration_digest

    # Set API key / base_url as environment variables
    backend = agent.get("backend", "openai")
    api_key = agent.get("api_key", "")
    if backend == "gemini":
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
        os.environ.setdefault("GEMINI_API_KEY", api_key)
    else:
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
        base_url = agent.get("base_url", "")
        if base_url:
            os.environ.setdefault("OPENAI_API_BASE", base_url)

    ns = argparse.Namespace(
        model_name=agent.get("model_name", "gpt-4.1-mini"),
        memory_system=memory.get("memory_system_name", "none"),
        memory_timeout=int(memory.get("timeout", 300)),
        use_step_memory=memory.get("use_step_memory", False),
        server_url=memory.get("server_url", "http://0.0.0.0:8000"),
        env_server_url=env.get("env_server_url", "http://0.0.0.0:8001"),
        judgement_mode=env.get("env_config", {}).get("judgement_mode", "none"),
        max_tokens=int(agent.get("max_tokens", 8192)),
        max_steps=task_specific.get("max_steps", 30),
        output_dir=output.get("output_dir", "./"),
        log_dir=output.get("log_dir", ""),
        global_csv=output.get("global_csv", ""),
        start_index=int(task_specific.get("start_index", 0) or 0),
        end_index=task_specific.get("end_index"),
        max_tasks=task_specific.get("max_tasks"),
        task_ids=task_specific.get("task_ids"),
        strict_plan_submission=bool(task_specific.get("strict_plan_submission", False)),
        strict_submission_attempts=int(task_specific.get("strict_submission_attempts", 3)),
        strict_search_step_limit=int(task_specific.get("strict_search_step_limit", 6)),
        strict_submission_max_tokens=int(task_specific.get("strict_submission_max_tokens", 4096)),
        independent_plan_capture=bool(task_specific.get("independent_plan_capture", False)),
        run_id=benchmark_contract.get("run_id"),
        config_digest=configuration_digest(cfg),
    )
    return ns


def _run_travel_sample(args, agent, query_data, output_file):
    data_idx = query_data["id"]
    task_id = str(uuid.uuid4())
    env_client = None
    memory_system = None
    completed = False
    primary_error = None
    cleanup_errors = []
    failure_stage = "initialize"
    all_results = []
    all_scratchpads = []
    all_round_results = []
    final_parsed_results = []

    try:
        env_client = EnvironmentClient(
            task_id=task_id,
            env_name="travel_planner",
            base_url=args.env_server_url,
            env_config={"judgement_mode": args.judgement_mode},
        )
        print(f"[Data {data_idx}] Environment task_id: {task_id}")
        failure_stage = "environment_reset"
        obs = env_client.reset(seed=data_idx)

        failure_stage = "memory_initialize"
        memory_user_id = f"data_{data_idx}_{args.model_name}_{args.memory_system}"
        memory_system = get_memory_system(
            args.memory_system,
            memory_user_id,
            args.server_url,
            timeout=args.memory_timeout,
            run_id=args.run_id,
            config_digest=args.config_digest,
        )
        if memory_system:
            print(f"\n[Memory] Initialized {args.memory_system} for user_id: {memory_user_id}")

        multi_queries = obs["questions"]
        ground_truth_list = obs.get("answers", [])
        gt_by_round = {gt["round_idx"]: gt for gt in ground_truth_list}
        total_rounds = len(multi_queries)
        accumulated_plan_text = ""
        latest_plans = {}
        round_judgements = {}
        expected_days = None
        subquery_times = {}
        agent.reset()

        base_person = obs.get("base_person")
        if base_person:
            base_name = base_person["name"]
            base_query = base_person["query"]
            base_plan = format_person_plan(base_name, base_person["daily_plans"])
            expected_days = len(base_person["daily_plans"])
            accumulated_plan_text = base_plan
            print(f"[Data {data_idx}] Base person set: {base_name}")
            if memory_system:
                failure_stage = "base_memory_add"
                base_chunk = json.dumps({
                    "name": base_name,
                    "query": base_query,
                    "is_base_person": True,
                    "final_plan": base_plan,
                }, ensure_ascii=False)
                memory_system.add(base_chunk)
                if args.memory_system in ["mem0", "mem0-g"]:
                    print("[Memory] Waiting 10s for Mem0 indexing...")
                    time.sleep(50)
                print(f"[Memory] Added base person to memory ({len(base_chunk)} chars)")
            else:
                agent.set_base_person(base_name, base_query, base_plan)

        for round_item in multi_queries:
            subquery_start = time.time()
            single_query = round_item["query"]
            round_idx = round_item["round_idx"]
            name = round_item.get("name", f"User{round_idx}")
            failure_stage = f"round_{round_idx}"

            print(f"\n{'=' * 50}")
            print(f"[Data {data_idx}, Round {round_idx}, Name: {name}] Starting Agent...")
            print(f"{'=' * 50}")

            memory_context = None
            if memory_system:
                wrapped_prompt = memory_system.wrap_user_prompt(single_query)
                memory_context = wrapped_prompt.split("</memory_context>")[0] + "</memory_context>"
                memory_context = memory_context.strip()

            use_step_memory = getattr(args, "use_step_memory", False)
            agent.prepare_for_person(
                name=name,
                round_idx=round_idx,
                include_previous_plans=(memory_system is None),
                memory_context=memory_context,
                memory_system=memory_system if use_step_memory else None,
                expected_days=expected_days,
            )
            action = agent.act(single_query)
            if getattr(args, "strict_plan_submission", False):
                if not agent.last_result.success or not action:
                    raise RuntimeError(
                        f"Data {data_idx} round {round_idx} strict plan failure: "
                        f"{agent.last_result.error_message}"
                    )

            accumulated_plan_text += f"\n\n{action}" if accumulated_plan_text else action
            agent_result = agent.last_result
            gt = gt_by_round.get(round_idx)
            if gt and "daily_plans" in gt:
                ground_truth = {
                    "name": name,
                    "daily_plans": gt["daily_plans"],
                    "judgement_mode": args.judgement_mode,
                }
                result = env_client.step(action, ground_truth=ground_truth, need_judge=True)
            else:
                result = env_client.step(action)

            observation = result["observation"]
            reward = result["reward"]
            info = result.get("info", {})
            all_round_results.append({
                "round": round_idx,
                "name": name,
                "query": single_query,
                "result": action,
            })

            scratchpad_dict = agent.get_scratchpad_dict()
            all_scratchpads.append({
                "round": round_idx,
                "name": name,
                "scratchpad": scratchpad_dict,
                "total_steps": agent_result.total_steps,
                "success": agent_result.success,
                "error_message": agent_result.error_message,
            })

            if getattr(args, "independent_plan_capture", False):
                parsed_action = parse_all_plans(action, multi_queries[:round_idx])
                for parsed in parsed_action:
                    if parsed.get("result"):
                        latest_plans[parsed["name"]] = parsed
                if name not in latest_plans:
                    raise RuntimeError(
                        f"Data {data_idx} round {round_idx} omitted current traveler {name}"
                    )
                parsed_this_round = []
                for idx, query_item in enumerate(multi_queries[:round_idx], start=1):
                    query_name = query_item.get("name", f"Person{idx}")
                    parsed = latest_plans.get(query_name)
                    parsed_this_round.append({
                        "person_idx": idx,
                        "name": query_name,
                        "query": query_item["query"],
                        "result": parsed["result"] if parsed else "",
                    })
            else:
                parsed_this_round = parse_all_plans(
                    accumulated_plan_text, multi_queries[:round_idx]
                )
            all_results.append(parsed_this_round)

            judgement = info.get("judgement")
            if judgement:
                round_judgements[round_idx] = judgement

            if memory_system:
                obs_with_judgement = {**(observation or {}), "judgement": judgement}
                memory_entry = agent.build_memory_entry(
                    task=single_query,
                    action=action,
                    observation=obs_with_judgement,
                    reward=reward,
                )
                memory_system.add(memory_entry)
                if args.memory_system in ["mem0", "mem0-g"]:
                    print("[Memory] Waiting for Mem0 indexing...")
                    time.sleep(60)

            if round_idx < total_rounds and memory_system is None:
                all_feedback = [
                    round_judgements[r]
                    for r in sorted(round_judgements)
                    if r <= round_idx
                ]
                if all_feedback:
                    agent.add_judge_feedback("\n\n".join(all_feedback))

            subquery_times[round_idx] = time.time() - subquery_start

        failure_stage = "result_validation"
        final_parsed_results = all_results[-1] if all_results else []
        for person in final_parsed_results:
            person_idx = person.get("person_idx")
            if person_idx is not None and person_idx in subquery_times:
                person["subquery_time"] = round(subquery_times[person_idx], 2)

        result_to_save = {
            "metadata": {
                "model_name": args.model_name,
                "judgement_mode": args.judgement_mode,
                "memory_system": args.memory_system,
                "mode": "agent",
                "max_steps": args.max_steps,
                "strict_plan_submission": getattr(args, "strict_plan_submission", False),
                "strict_submission_max_tokens": getattr(args, "strict_submission_max_tokens", 4096),
                "independent_plan_capture": getattr(args, "independent_plan_capture", False),
                "terminal_failure": False,
            },
            "status": "completed",
            "failure": None,
            f"{args.model_name}_sole-planning_results": final_parsed_results,
            "all_results": all_results,
            "scratchpads": all_scratchpads,
        }
        _validate_travel_success_payload(result_to_save, args.model_name, total_rounds)
        failure_stage = "result_write"
        write_json_atomic(output_file, result_to_save)
        completed = True
    except Exception as error:
        primary_error = error
    finally:
        if env_client is not None:
            try:
                env_client.close()
            except Exception as error:
                cleanup_errors.append(error)
        if memory_system is not None:
            try:
                memory_system.close(completed=completed and primary_error is None)
            except Exception as error:
                cleanup_errors.append(error)

    if cleanup_errors:
        cleanup_error = RuntimeError(
            "; ".join(_bounded_error_message(error) for error in cleanup_errors)
        )
        if primary_error is None:
            primary_error = cleanup_error
            failure_stage = "cleanup"

    if primary_error is not None:
        failure_payload = _build_travel_failure_payload(
            args,
            data_idx,
            primary_error,
            failure_stage,
            all_results,
            all_scratchpads,
        )
        write_json_atomic(output_file, failure_payload)
        raise RuntimeError(
            f"Travel data {data_idx} failed at {failure_stage}: "
            f"{_bounded_error_message(primary_error)}"
        ) from primary_error

    return result_to_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON config file (replaces all other args)")
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--judgement_mode", type=str, default="hint",
                        choices=["answer", "hint", "none"])
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--strict_submission_max_tokens", type=int, default=4096)
    parser.add_argument("--memory_system", type=str, default="none",
                        choices=["none", "long_context", "mirix", "letta", "mem0", "mem0-g",
                                 "rag", "bm25", "text-embedding-3-small", "cognee", "memorag",
                                 "graphrag", "lightmem", "amem", "amem_indepth",
                                 "memorybank_indepth", "reasoningbank", "zep"])
    parser.add_argument("--server_url", type=str, default="http://0.0.0.0:8000")
    parser.add_argument("--memory_timeout", type=int, default=300)
    parser.add_argument("--env_server_url", type=str, default="http://0.0.0.0:8001")
    raw_args = parser.parse_args()

    if raw_args.config:
        args = load_config(raw_args.config)
    else:
        args = raw_args

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = getattr(args, 'log_dir', '') or ''
    global_csv = getattr(args, 'global_csv', '') or ''
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("SOLE PLANNING - Agent Mode")
    print("=" * 60)
    print(f"Model:         {args.model_name}")
    print(f"Data source:   HuggingFace (ZexueHe/memoryarena, group_travel_planner)")
    print(f"Output dir:    {args.output_dir}")
    print(f"Judgement:     {args.judgement_mode}")
    print(f"Memory system: {args.memory_system}")
    print(f"Memory URL:    {args.server_url}")
    print(f"Env URL:       {args.env_server_url}")
    print(f"Max tokens:    {args.max_tokens}")
    print(f"Max steps:     {args.max_steps}")
    print("=" * 60 + "\n")

    query_data_list = load_travel_data()
    start_index = int(getattr(args, "start_index", 0) or 0)
    end_index = getattr(args, "end_index", None)
    if end_index is None:
        max_tasks = getattr(args, "max_tasks", None)
        end_index = len(query_data_list) if max_tasks is None else start_index + int(max_tasks)
    end_index = min(len(query_data_list), int(end_index))
    task_ids = getattr(args, "task_ids", None)
    if task_ids is not None:
        requested_ids = [int(item) for item in task_ids]
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("task_specific.task_ids contains duplicates")
        by_id = {int(item["id"]): item for item in query_data_list}
        missing = sorted(set(requested_ids) - set(by_id))
        if missing:
            raise ValueError(f"Unknown Travel task IDs: {missing}")
        query_data_list = [by_id[item] for item in requested_ids]
    else:
        query_data_list = query_data_list[start_index:end_index]
    print(f"Loaded {len(query_data_list)} data items")
    if query_data_list:
        print('First item keys:', list(query_data_list[0].keys()))

    agent = TravelPlannerAgent(
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
        strict_plan_submission=getattr(args, "strict_plan_submission", False),
        strict_submission_attempts=getattr(args, "strict_submission_attempts", 3),
        strict_search_step_limit=getattr(args, "strict_search_step_limit", 6),
        strict_submission_max_tokens=getattr(args, "strict_submission_max_tokens", 4096),
    )

    start_time = time.time()
    aggregate_failures = []

    for query_data in tqdm(query_data_list):
        data_idx = query_data["id"]
        output_file = os.path.join(args.output_dir, f"generated_plan_{data_idx}.json")
        existing = None
        if os.path.exists(output_file):
            existing = _load_existing_travel_result(
                output_file,
                args.model_name,
                len(query_data.get("questions", [])),
            )
        if existing is not None:
            print(f"[Data {data_idx}] Existing terminal output found, skipping: {output_file}")
            if existing.get("status") == "failed":
                aggregate_failures.append(
                    f"data {data_idx}: existing terminal failure"
                )
            continue
        if os.path.exists(output_file):
            message = (
                f"data {data_idx}: existing non-terminal artifact is preserved; "
                "use a fresh output directory"
            )
            print(f"[Data {data_idx}] {message}: {output_file}")
            aggregate_failures.append(message)
            continue

        try:
            _run_travel_sample(args, agent, query_data, output_file)
        except Exception as error:
            aggregate_failures.append(
                f"data {data_idx}: {_bounded_error_message(error)}"
            )
            print(f"[Data {data_idx}] FAILED: {_bounded_error_message(error)}")
        else:
            print(f"[Data {data_idx}] Completed: {output_file}")

    if aggregate_failures:
        raise RuntimeError(
            f"{len(aggregate_failures)} travel sample(s) reached a failed terminal state: "
            + " | ".join(aggregate_failures)
        )

    # --- Usage stats ---
    duration_seconds = time.time() - start_time
    duration_minutes = duration_seconds / 60

    usage = agent.get_usage_stats()
    usage['duration_seconds'] = round(duration_seconds, 2)
    usage['duration_minutes'] = round(duration_minutes, 2)

    print(f"\n=== Usage Stats ===")
    print(f"Total input tokens: {usage['total_input_tokens']}")
    print(f"Total output tokens: {usage['total_output_tokens']}")
    print(f"Total cost: ${usage['total_cost']:.4f}")
    print(f"Duration: {duration_minutes:.2f} minutes ({duration_seconds:.2f} seconds)")

    stats_dir = os.path.join(args.output_dir, "stats_results")
    os.makedirs(stats_dir, exist_ok=True)
    usage_file = os.path.join(stats_dir, "usage_stats.json")
    write_json_atomic(usage_file, usage)
    print(f"Usage stats saved to: {usage_file}")

    # --- Postprocess: combination + eval ---
    submission_dir = os.path.join(args.output_dir, "submission")

    print("\n--- Running combination ---")
    submission_file = combine(
        model_name=args.model_name,
        output_dir=args.output_dir,
        submission_file_dir=submission_dir,
        mode="sole_planning",
    )

    print("\n--- Running evaluation ---")
    evaluate(
        submission_path=submission_file,
        model_name=args.model_name,
        memory_system=args.memory_system,
        global_csv=global_csv,
    )

    print("Done!")


if __name__ == "__main__":
    main()
