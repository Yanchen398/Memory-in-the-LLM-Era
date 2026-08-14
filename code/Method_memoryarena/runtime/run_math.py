"""
Complete integration example showing Memory + Environment working together.

This script demonstrates the exact workflow from your pseudo-code:
1. Create task_id
2. Initialize memory client
3. Initialize environment client
4. Run task loop with memory-wrapped prompts
5. Store experiences in memory
"""

import argparse
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any
from env.env_client import EnvironmentClient
import pdb
import time
import os
import sys
import json

_CODE_ROOT = Path(__file__).resolve().parents[2]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from Method_memoryarena.client import MemoryClient, configuration_digest
from env.env_systems.formal_reasoning_env.eval import eval_and_print_result
from agent import MathAgent
from datasets import load_dataset
from dataclasses import asdict

logger = logging.getLogger(__name__)


def _safe_path_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return safe[:180] or "paper"


def _save_logs_jsonl(logs: List[Dict[str, Any]], json_output_file: str) -> None:
    """Atomically save logs as JSONL with backward-compatible fields."""
    print(f"Saving Logs.... len={len(logs)}")
    output_dir = os.path.dirname(json_output_file)
    os.makedirs(output_dir, exist_ok=True)
    tmp_file = os.path.join(
        output_dir,
        f".{os.path.basename(json_output_file)}.{uuid.uuid4().hex}.tmp",
    )
    with open(tmp_file, "w", encoding="utf-8") as jsonf:
        for log in logs:
            log_dict = log.copy()
            if 'judge_result' in log_dict and hasattr(log_dict['judge_result'], '__dict__'):
                log_dict['judge_result'] = asdict(log_dict['judge_result'])
            jsonf.write(json.dumps(log_dict, ensure_ascii=False) + "\n")
        jsonf.flush()
        os.fsync(jsonf.fileno())
    os.replace(tmp_file, json_output_file)


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    output_dir = os.path.dirname(path)
    os.makedirs(output_dir, exist_ok=True)
    tmp_file = os.path.join(
        output_dir,
        f".{os.path.basename(path)}.{uuid.uuid4().hex}.tmp",
    )
    with open(tmp_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_file, path)


def _bounded_error_message(error: Exception, limit: int = 8000) -> str:
    message = f"{type(error).__name__}: {error}"
    if len(message) <= limit:
        return message
    half = max(1, (limit - 80) // 2)
    return (
        message[:half]
        + "\n...[terminal error evidence truncated]...\n"
        + message[-half:]
    )


def _classify_failure(message: str) -> str:
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


def _build_terminal_failure_rows(
    tasks: List[Any],
    completed_logs: List[Dict[str, Any]],
    error: Exception,
    stage: str,
) -> List[Dict[str, Any]]:
    """Preserve completed queries and represent every remaining query as failed."""
    error_message = _bounded_error_message(error)
    failure_kind = _classify_failure(error_message)
    rows = list(completed_logs)
    if rows and len(rows) >= len(tasks):
        # A validation/write failure after the final query must still leave an
        # explicit failed row rather than an apparently successful paper.
        rows = rows[: len(tasks) - 1]
    for query_id in range(len(rows), len(tasks)):
        subtask, ground_truth, _background = tasks[query_id]
        rows.append({
            "metadata": {
                "terminal_failure": True,
                "failure_kind": failure_kind,
                "failure_message": error_message,
                "failure_policy": "record_as_incorrect_and_continue",
            },
            "status": "failed",
            "query_id": query_id,
            "query": subtask,
            "memory_context": None,
            "response": "",
            "is_correct": False,
            "time": 0.0,
            "ground_truth": ground_truth,
            "judge_result": {
                "feedback": None,
                "is_correct": False,
                "status": "not_run_agent_failure",
            },
            "observation": None,
            "failure": {
                "kind": failure_kind,
                "stage": stage,
                "message": error_message,
            },
        })
    return rows


def _validate_success_logs(logs: List[Dict[str, Any]], expected_count: int) -> None:
    if len(logs) != expected_count:
        raise ValueError(f"Expected {expected_count} query rows, got {len(logs)}")
    for expected_id, row in enumerate(logs):
        if row.get("query_id") != expected_id:
            raise ValueError(f"Unexpected query_id at row {expected_id}: {row.get('query_id')}")
        if row.get("status") != "completed":
            raise ValueError(f"Query {expected_id} is not completed")
        if not isinstance(row.get("response"), str) or not row["response"].strip():
            raise ValueError(f"Query {expected_id} has an empty response")
        if not isinstance(row.get("is_correct"), bool):
            raise ValueError(f"Query {expected_id} has non-boolean is_correct")


def _read_paper_rows(result_file: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(result_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _get_json_output_dir(cfg: Dict[str, Any]) -> str:
    """Return the method-specific JSON output directory for this run."""
    memory_name = cfg["memory"]["memory_system_name"]
    output_root = cfg["output"].get("json_output_dir") or cfg["output"].get("output_dir")
    if not output_root:
        raise KeyError("output.json_output_dir or output.output_dir is required")
    if memory_name != "long_context":
        return os.path.join(output_root, memory_name)
    return os.path.join(
        output_root,
        f"{memory_name}_{cfg['agent']['model_name']}",
    )


def _get_paper_result_file(cfg: Dict[str, Any], paper_key: str) -> str:
    """Return the result file path for a paper."""
    json_path = _get_json_output_dir(cfg)
    if paper_key:
        return os.path.join(json_path, paper_key, "result.jsonl")
    return os.path.join(json_path, "result.jsonl")


def _is_paper_processed(
    cfg: Dict[str, Any],
    paper_key: str,
    expected_num_queries: int,
) -> bool:
    """Check whether a paper already has a complete result file."""
    result_file = _get_paper_result_file(cfg, paper_key)
    if not os.path.isfile(result_file):
        return False

    if expected_num_queries <= 0:
        return os.path.getsize(result_file) > 0

    try:
        rows = _read_paper_rows(result_file)
    except (OSError, json.JSONDecodeError):
        return False
    if len(rows) != expected_num_queries:
        return False
    for query_id, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("query_id") != query_id:
            return False
        if row.get("status", "completed") not in {"completed", "failed"}:
            return False
    return True


def _paper_has_terminal_failure(cfg: Dict[str, Any], paper_key: str) -> bool:
    result_file = _get_paper_result_file(cfg, paper_key)
    if os.path.isfile(f"{result_file}.terminal_failure.json"):
        return True


def _load_reasoning_dataset(cfg: Dict[str, Any]):
    dataset_cfg = cfg["task_specific"]["dataset"]
    local_path = dataset_cfg.get("local_path")
    local_only = bool((cfg.get("benchmark_contract") or {}).get("local_only"))
    if local_path:
        path = os.path.abspath(os.path.expanduser(str(local_path)))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Local reasoning dataset not found: {path}")
        suffix = os.path.splitext(path)[1].lower()
        if suffix not in {".json", ".jsonl"}:
            raise ValueError(
                f"Local reasoning dataset must be JSON or JSONL, got: {path}"
            )
        return load_dataset("json", data_files=path, split="train")
    if local_only:
        raise ValueError(
            "benchmark_contract.local_only requires task_specific.dataset.local_path"
        )
    return load_dataset(
        dataset_cfg["hf_dataset"],
        dataset_cfg["hf_config"],
        split=dataset_cfg["hf_split"],
    )
    try:
        return any(
            row.get("status") == "failed"
            or bool((row.get("metadata") or {}).get("terminal_failure"))
            for row in _read_paper_rows(result_file)
        )
    except (OSError, json.JSONDecodeError):
        return True


def run_task_with_memory_and_env(
    cfg: Dict[str, Any] ,
    tasks: List[str],
    paper_key: str,
):
    """
    Run a complete task loop with memory and environment integration.
    
    This follows your exact pseudo-code pattern:
    + task_id = uuid.uuid4()
    + client = MemoryClient(task_id, memory_system_name='mirix')
     
     obs = get_initial_obs(env)
     for i in range(task_num):
    +    prompt = client.wrap_user_prompt(task_i)
    +    action = agent(prompt)
         obs = env(action)
    +    client.add(f"action: {action}\\nobs: {obs}")
    """
    logger.info("\n%s", "=" * 80)
    logger.info("RUNNING TASK: %s", cfg["env"]["env_name"].upper())
    logger.info("%s", "=" * 80)

    if _is_paper_processed(cfg, paper_key, len(tasks)):
        logger.info(
            "Paper %s already has a complete result file. Skipping.",
            paper_key,
        )
        return []
    
    if not tasks:
        raise ValueError(f"Paper {paper_key} has no tasks")

    task_id = str(uuid.uuid4())
    json_output_file = _get_paper_result_file(cfg, paper_key)
    if os.path.exists(json_output_file):
        raise FileExistsError(
            "Existing non-terminal paper artifact is preserved; use a fresh output root: "
            f"{json_output_file}"
        )
    logs: List[Dict[str, Any]] = []
    memory_client = None
    env_client = None
    completed = False
    primary_error = None
    failure_stage = "initialize"
    cleanup_errors: List[Exception] = []

    try:
        agent = MathAgent(
            model_name=cfg["agent"]["model_name"],
            temperature=cfg["agent"]["temperature"],
            max_tokens=cfg["agent"]["max_tokens"],
            backend=cfg["agent"]["backend"],
            base_url=cfg["agent"]["base_url"] if cfg["agent"]["backend"] == "openai" else None,
        )
        failure_stage = "memory_initialize"
        memory_client = MemoryClient(
            task_id,
            memory_system_name=cfg["memory"]["memory_system_name"],
            base_url=cfg["memory"]["base_url"],
            timeout=cfg["memory"]["timeout"],
            run_id=(cfg.get("benchmark_contract") or {}).get("run_id"),
            config_digest=configuration_digest(cfg),
        )
        failure_stage = "environment_initialize"
        env_client = EnvironmentClient(
            task_id=task_id,
            env_name=cfg["env"]["env_name"],
            base_url=cfg["env"]["base_url"],
            timeout=cfg["env"]["timeout"],
            env_config=cfg["env"]["env_config"] or {"max_steps": 10},
        )
        env_client.reset()

        for subtask_idx, (subtask, ground_truth, background) in enumerate(tasks):
            failure_stage = f"query_{subtask_idx}"
            t0 = time.time()
            query = agent.build_prompt(task=subtask, background=background)
            prompt = memory_client.wrap_user_prompt(query)
            action = agent.act(prompt)

            logger.info("\n--- Agent -> Env ---")
            logger.info("Memory-wrapped prompt: %s", prompt)
            logger.info("Agent action: %s", action)

            result = env_client.step(action, ground_truth=ground_truth, need_judge=True)
            observation = result.get("observation") or {}
            response = observation.get("final")
            reward = result.get("reward")
            is_correct = bool(reward) if reward is not None else None
            logger.info("--- Env -> Agent ---")
            logger.info("Env response: %s", response)
            if reward is not None:
                logger.info("Judge reward: %s", reward)

            memory_entry = agent.build_memory_entry(
                task=subtask,
                observation=observation,
                action=action,
                reward=(
                    reward
                    if cfg["memory"].get("judge_result_in_memory", False)
                    and result.get("judge_result") is not None
                    else None
                ),
            )
            memory_client.add(memory_entry)
            logs.append({
                "metadata": {"terminal_failure": False},
                "status": "completed",
                "failure": None,
                "query_id": subtask_idx,
                "query": subtask,
                "memory_context": observation.get("memory_context"),
                "response": response,
                "is_correct": is_correct,
                "time": time.time() - t0,
                "ground_truth": ground_truth,
                "judge_result": {
                    "feedback": observation.get("judge_result"),
                    "is_correct": is_correct,
                    "status": "completed",
                },
                "observation": observation,
            })

        failure_stage = "result_validation"
        _validate_success_logs(logs, len(tasks))
        failure_stage = "result_write"
        _save_logs_jsonl(logs, json_output_file)
        completed = True
        logger.info("Saved %s logs to: %s", len(logs), json_output_file)
    except Exception as error:
        primary_error = error
        terminal_rows = _build_terminal_failure_rows(tasks, logs, error, failure_stage)
        try:
            _save_logs_jsonl(terminal_rows, json_output_file)
        except Exception as artifact_error:
            cleanup_errors.append(artifact_error)
    finally:
        if env_client is not None:
            try:
                env_client.close()
            except Exception as error:
                cleanup_errors.append(error)
        if memory_client is not None:
            try:
                memory_client.close(completed=completed and primary_error is None)
            except Exception as error:
                cleanup_errors.append(error)

    if cleanup_errors:
        cleanup_error = RuntimeError(
            "; ".join(_bounded_error_message(error) for error in cleanup_errors)
        )
        _write_json_atomic(
            f"{json_output_file}.terminal_failure.json",
            {
                "status": "failed",
                "paper_key": paper_key,
                "failure": {
                    "kind": "cleanup_error",
                    "stage": "cleanup",
                    "message": _bounded_error_message(cleanup_error),
                },
            },
        )
        if primary_error is None:
            primary_error = cleanup_error

    if primary_error is not None:
        raise RuntimeError(
            f"Paper {paper_key} failed at {failure_stage}: {_bounded_error_message(primary_error)}"
        ) from primary_error
    return logs


def main(json_config):
    """Run examples for all three environments."""
    logger.info("\n%s", "=" * 80)
    logger.info("MEMORY + ENVIRONMENT INTEGRATION DEMONSTRATION")
    logger.info("%s", "=" * 80)
    logger.info("\nThis demonstrates the complete workflow:")
    logger.info("1. Initialize memory and environment clients")
    logger.info("2. Wrap tasks with memory context")
    logger.info("3. Agent generates actions based on memory")
    logger.info("4. Store experiences back to memory")
    logger.info("\nNote: Make sure env_server.py is running on port 8001")
    
 
    ds = _load_reasoning_dataset(json_config)
    
    """
    ds[0].keys()
    dict_keys(['id', 'paper_name', 'questions', 'answers', 'backgrounds'])
    """
    task_cfg = json_config.get("task_specific", {})
    start_index = int(task_cfg.get("start_index", 0) or 0)
    end_index = task_cfg.get("end_index")
    if end_index is None:
        max_tasks = task_cfg.get("max_tasks")
        end_index = len(ds) if max_tasks is None else start_index + int(max_tasks)
    end_index = min(len(ds), int(end_index))

    aggregate_failures = []
    for i, paper_id in enumerate(range(start_index, end_index)):
        
        paper_name=ds[paper_id]['paper_name']
        paper_key = _safe_path_component(paper_name)
      
        
        tasks=[
            (ds[paper_id]['questions'][i], ds[paper_id]['answers'][i], ds[paper_id]['backgrounds'][i]) for i in range(len(ds[paper_id]['questions']))
        ]

        if _is_paper_processed(json_config, paper_key, len(tasks)):
            logger.info(
                "Skipping already processed paper %s (%s/%s).",
                paper_name,
                paper_id + 1,
                len(ds),
            )
            if _paper_has_terminal_failure(json_config, paper_key):
                aggregate_failures.append(f"{paper_key}: existing terminal failure")
            continue
        
        try:
            run_task_with_memory_and_env(
                cfg=json_config,
                tasks=tasks,
                paper_key=paper_key,
            )
        except Exception as error:
            logger.exception("Paper %s failed", paper_name)
            aggregate_failures.append(f"{paper_key}: {_bounded_error_message(error)}")
            continue

        logger.info("\n%s", "=" * 80)
        logger.info("PAPER %s COMPLETED", paper_name)
        logger.info("%s", "=" * 80)
    logger.info("All papers completed.")

    if aggregate_failures:
        raise RuntimeError(
            f"{len(aggregate_failures)} paper(s) reached a failed terminal state: "
            + " | ".join(aggregate_failures)
        )
    
    #===============EVAL=================
    if json_config["task_specific"].get("auto_eval_after_run"):
        logger.info("Start evaluating results...")
        # eval_and_print_result(json_config, logger)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # use argparse to parse command line arguments for json config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the JSON config file", default="configs/formal_reasoning_configs/math_longcontext_gpt-5-mini.json")
    args = parser.parse_args()

    # take the first arg from command line as json_config, if not  provided, raies warning and say "input json config`"
    
    json_config = json.load(open(args.config))
    try:
        main(json_config)
    except Exception as e:
        logger.exception("❌ Error while running example: %s", e)
        logger.error("Troubleshooting:")
        logger.error("1. Make sure the environment server is running: python env_server.py")
        logger.error("2. Check that the server is accessible at http://0.0.0.0:8001")
        logger.error("3. If using real MemoryClient, ensure memory server is running at http://0.0.0.0:8000")
        raise
