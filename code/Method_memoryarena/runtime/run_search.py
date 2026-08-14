"""
Run BrowseComp-Plus search via the environment server.

This script calls the env server (env/env_server.py, port 8001) and memory server (memory/server.py, port 8000),
runs the search agent for each query, and returns the results.

Usage (from project root, env server already running via `python env/env_server.py`):

    python run_search.py --config configs/web_search_configs/search_task.json

All settings (including query_ids) are loaded from the config JSON file.
"""

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

# Project root = directory containing run_search.py (no "MemoryArena" package name)
_SCRIPT_DIR = Path(__file__).resolve().parent
# Load .env so OPENAI_API_KEY and OPENAI_BASE_URL are set (optional: python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv(_SCRIPT_DIR / ".env", override=True)
except ImportError:
    pass
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_ENV_SYSTEMS = _SCRIPT_DIR / "env" / "env_systems"

def _setup_paths():
    # So "agent" and "env" resolve to agent/ and env/ (no MemoryArena package)
    if str(_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPT_DIR))
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    if str(_ENV_SYSTEMS) not in sys.path:
        sys.path.insert(0, str(_ENV_SYSTEMS))

_setup_paths()

from agent.search import load_correct_answers
from Method_memoryarena.client import configuration_digest


def load_config(config_path: Path) -> dict:
    """Load JSON config file and return as dict."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write one completed task artifact without exposing a partial JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


class ConfigArgs:
    """Simple namespace to hold config values."""
    pass


def config_to_args(config: dict) -> ConfigArgs:
    """Convert config dict to args-like namespace."""
    args = ConfigArgs()
    contract = config.get("benchmark_contract", {})
    args.run_id = contract.get("run_id")
    args.config_digest = configuration_digest(config)
    
    # Agent settings
    agent_cfg = config.get("agent", {})
    args.model_name = agent_cfg.get("model_name", "gpt-5-mini")
    args.embedding_model = agent_cfg.get("embedding_model", "text-embedding-3-small")
    args.max_tokens = int(agent_cfg.get("max_tokens", 8192))
    # Eight Search calls plus one forced-answer turn.
    args.max_iterations = int(agent_cfg.get("max_iterations", 9))
    args.retrieval_top_k = int(agent_cfg.get("retrieval_top_k", 10))
    args.snippet_max_tokens = int(agent_cfg.get("snippet_max_tokens", 64))
    args.max_search_calls = int(agent_cfg.get("max_search_calls", 8))
    args.provider = agent_cfg.get("provider")
    args.api_key = agent_cfg.get("api_key", "")
    args.base_url = agent_cfg.get("base_url", "")

    # Memory settings
    mem_cfg = config.get("memory", {})
    args.memory_system = mem_cfg.get("memory_system_name", "bm25")
    args.memory_url = mem_cfg.get("memory_url", "http://0.0.0.0:8000")
    args.memory_timeout = int(mem_cfg.get("timeout", 300))
    args.no_memory = mem_cfg.get("no_memory", False)
    args.step_memory = mem_cfg.get("use_step_memory", False)

    # Env settings
    env_cfg = config.get("env", {})
    args.env_server_url = env_cfg.get("env_server_url", "http://0.0.0.0:8001")
    args.script_path = env_cfg.get("script_path")
    args.searcher_type = env_cfg.get("searcher_type", "openai")
    args.index_path = env_cfg.get("index_path", "env/env_systems/web_search_env/embeddings/shard*.index")
    args.corpus_path = env_cfg.get("corpus_path", "web_search_env/data/corpus.jsonl")
    args.gpu = env_cfg.get("gpu")
    args.mcp_url = env_cfg.get("mcp_url")
    args.timeout = env_cfg.get("timeout", 3000)
    args.mcp_name = env_cfg.get("mcp_name", "retrieval-mcp-server")

    # Task-specific settings
    task_cfg = config.get("task_specific", {})
    args.query_ids = task_cfg.get("query_ids", [])
    args.data_dir = Path(task_cfg.get("data_dir", "env/env_systems/web_search_env/data"))
    args.qrel_evidence = task_cfg.get("qrel_evidence", "topics-qrels/qrel_evidence.txt")
    args.judge_model = task_cfg.get("judge_model", "gpt-4.1")
    args.judge_protocol = task_cfg.get(
        "judge_protocol", "official_provider"
    )
    args.store_eval_in_memory = task_cfg.get("store_eval_in_memory", False)
    args.defer_judge_if_unavailable = bool(
        task_cfg.get("defer_judge_if_unavailable", False)
    )
    args.record_query_errors_as_failures = bool(
        task_cfg.get("record_query_errors_as_failures", False)
    )
    judge_key_env = task_cfg.get(
        "judge_api_key_env", "MEMORYARENA_OFFICIAL_JUDGE_API_KEY"
    )
    judge_base_env = task_cfg.get(
        "judge_base_url_env", "MEMORYARENA_OFFICIAL_JUDGE_BASE_URL"
    )
    args.judge_api_key = task_cfg.get("judge_api_key") or os.getenv(
        judge_key_env
    )
    args.judge_base_url = task_cfg.get("judge_base_url") or os.getenv(
        judge_base_env
    )

    # Output settings
    out_cfg = config.get("output", {})
    args.output_dir = Path(out_cfg.get("output_dir", "out/search_run"))

    return args

# Environment server client (optional)
try:
    from env.env_client import EnvironmentClient
except ImportError:
    EnvironmentClient = None


def load_ground_truth(jsonl_path: Path) -> dict:
    gt = {}
    if not jsonl_path.exists():
        return gt
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            qid = str(obj.get("query_id", ""))
            gt[qid] = {
                "query": obj.get("query", ""),
                "answer": obj.get("answer", ""),
                "evidence": obj.get("evidence_docs", obj.get("evidence", [])),
                "gold_docs": obj.get("gold_docs", []),
            }
    return gt


def load_qrel(qrel_path: Path) -> dict:
    """Load qrel (query id -> list of relevant docids) if available."""
    qrel = {}
    if not qrel_path or not qrel_path.exists():
        return qrel
    with qrel_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                qid, docid, rel = parts[0], parts[2], parts[-1]
                if rel != "0":
                    qrel.setdefault(qid, []).append(docid)
    return qrel


def load_qrel_data(qrel_path: Path) -> dict:
    """Load qrel evidence data for recall calculation: query_id -> list[doc_id]."""
    from collections import defaultdict

    qrel_data = defaultdict(list)
    if not qrel_path or not qrel_path.exists():
        return dict(qrel_data)

    with qrel_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                query_id = parts[0]
                doc_id = parts[2]
                qrel_data[query_id].append(doc_id)
    return dict(qrel_data)


def _summarize_single_result(out: dict, qrel_data: dict | None = None) -> dict:
    """
    Build an evaluation summary for a single result dict produced by the env server.

    The summary mirrors the human-readable lines printed at the end of main().
    """
    total = 1
    skipped = 0
    failed = 1 if out.get("status") == "failed" else 0

    judgement = out.get("judgement") or {}
    if failed:
        evaluated = 1
        accuracy = 0.0
        calibration_error = None
    elif not judgement or "correct" not in judgement:
        skipped = 1
        evaluated = 0
        accuracy = None
        calibration_error = None
    else:
        evaluated = 1
        correct = bool(judgement.get("correct"))
        confidence = judgement.get("confidence")
        acc_val = 1.0 if correct else 0.0
        accuracy = acc_val
        if isinstance(confidence, (int, float)):
            calibration_error = abs(confidence / 100.0 - acc_val)
        else:
            calibration_error = None

    # Recall: compute from retrieved_docids vs qrel for this query_id.
    # If there is no qrel entry or no relevant docs, recall is None.
    query_id = str(out.get("query_id") or "")
    avg_recall = None
    if qrel_data and query_id and query_id in qrel_data:
        relevant_docids = {str(d) for d in qrel_data[query_id]}
        if relevant_docids:
            retrieved = {str(d) for d in (out.get("retrieved_docids") or [])}
            if not retrieved:
                avg_recall = 0.0
            else:
                avg_recall = len(retrieved & relevant_docids) / len(relevant_docids)

    # Average tool calls per tool (for a single evaluation, just echo counts).
    tool_counts = out.get("tool_call_counts") or {}
    avg_tool_calls: dict[str, float] = {}
    if tool_counts:
        for tool, count in tool_counts.items():
            try:
                c = float(count)
            except Exception:
                continue
            avg_tool_calls[str(tool)] = c

    summary = {
        "processed_evaluations": total,
        "skipped_evaluations": skipped,
        "failed_evaluations": failed,
        # Every valid query occupies one evaluation slot. A missing judge is
        # still incorrect for Accuracy and contributes zero/missing tool calls.
        "evaluated_responses": total,
        "accuracy": accuracy,  # fraction in [0,1] or None
        "recall": avg_recall,  # fraction in [0,1] or None
        "average_tool_calls": avg_tool_calls,  # {tool_name: avg_calls}
        "calibration_error": calibration_error,  # fraction in [0,1] or None
    }
    return summary


def _bounded_error_message(error: Exception, limit: int = 8000) -> str:
    """Keep terminal failure evidence without embedding an entire subprocess log."""
    message = f"{type(error).__name__}: {error}"
    if len(message) <= limit:
        return message
    half = max(1, (limit - 80) // 2)
    return (
        message[:half]
        + "\n...[terminal error evidence truncated]...\n"
        + message[-half:]
    )


def _classify_query_failure(error_message: str) -> str:
    lowered = error_message.lower()
    if (
        "context length" in lowered
        or "maximum input length" in lowered
        or "input tokens" in lowered
    ):
        return "context_limit"
    if "timed out" in lowered or "timeout" in lowered:
        return "timeout"
    if "cuda" in lowered:
        return "cuda"
    if "http" in lowered or "server error" in lowered:
        return "http"
    return "agent_error"


def _build_terminal_failure_result(
    *,
    qid: str,
    correct_answer: str,
    judge_model: str,
    judge_protocol: str,
    error: Exception,
) -> dict:
    """
    Represent one failed official task as an explicit zero-score terminal result.

    This isolates independent queries without synthesizing an answer, compacting
    observations, or changing any successful agent/environment interaction.
    """
    error_message = _bounded_error_message(error)
    failure_kind = _classify_query_failure(error_message)
    return {
        "metadata": {
            "query_id": qid,
            "terminal_failure": True,
            "failure_kind": failure_kind,
            "failure_message": error_message,
            "failure_policy": "record_as_incorrect_and_continue",
        },
        "query_id": qid,
        "tool_call_counts": {},
        "usage": {},
        "status": "failed",
        "retrieved_docids": [],
        "result": [],
        "api_calls": [],
        "final_query": None,
        "predicted_answer": "",
        "correct_answer": correct_answer,
        "judgement": {
            "correct": False,
            "confidence": None,
            "reason": "Task agent did not produce a final answer.",
        },
        "judge_status": "not_run_agent_failure",
        "judge_model": judge_model,
        "judge_protocol": judge_protocol,
        "judge_error": None,
        "recall": 0.0,
        "failure": {
            "kind": failure_kind,
            "message": error_message,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run BrowseComp-Plus search (agent + browsecomp_plus env)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON config file (e.g. configs/web_search_configs/search_task.json)",
    )
    cli_args = parser.parse_args()

    # Load config and convert to args
    config_path = cli_args.config if cli_args.config.is_absolute() else (_SCRIPT_DIR / cli_args.config)
    config = load_config(config_path)
    args = config_to_args(config)
    args.query_id = args.query_ids  # from config
    if args.api_key:
        os.environ.setdefault("OPENAI_API_KEY", args.api_key)
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
        os.environ["OPENAI_API_BASE"] = args.base_url
    
    if not args.query_id:
        print("Error: No query_ids specified in config.")
        sys.exit(1)
    
    print(f"Loaded config from {config_path}")

    # Resolve paths
    data_dir = args.data_dir if args.data_dir.is_absolute() else (_SCRIPT_DIR / args.data_dir)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (_SCRIPT_DIR / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_path = data_dir / "browsecomp_plus_decrypted.jsonl"
    if not ground_truth_path.exists():
        print(f"Warning: ground truth not found at {ground_truth_path}")

    gt = load_ground_truth(ground_truth_path)
    query_ids = [str(q) for q in args.query_id]

    # Filter query IDs by availability in ground truth
    missing_ids = [qid for qid in query_ids if qid not in gt]
    for qid in missing_ids:
        print(f"Query ID {qid} not in ground truth. Skipping.")

    valid_ids = [qid for qid in query_ids if qid in gt]
    if not valid_ids:
        print(f"No valid query IDs found in ground truth. Available: {list(gt.keys())[:20]}...")
        sys.exit(1)

    script_path = args.script_path
    if script_path is None:
        script_path = str(
            _ENV_SYSTEMS / "web_search_env" / "search_agent" / "openai_client.py"
        )
    # Load qrel evidence for recall calculation; falls back to empty if not present.
    qrel_evidence_path = Path(args.qrel_evidence) if args.qrel_evidence else None
    qrel_data = load_qrel_data(qrel_evidence_path) if qrel_evidence_path else {}

    if EnvironmentClient is None:
        sys.exit(
            "EnvironmentClient not available. Install env dependency and ensure env/env_client.py is importable."
        )

    per_query_summaries = []
    per_query_results = {}

    for qid in valid_ids:
        query_result_file = output_dir / f"query_{qid}_result.json"
        if query_result_file.is_file():
            try:
                saved_payload = load_config(query_result_file)
                saved_item = (saved_payload.get("per_query") or {}).get(qid)
                saved_result = (saved_item or {}).get("result") or {}
                saved_summary = (saved_item or {}).get("summary") or {}
                if (
                    saved_payload.get("query_ids") == [qid]
                    and saved_result.get("query_id") == qid
                    and saved_result.get("status") in {"completed", "failed"}
                    and isinstance(saved_result.get("result"), list)
                    and isinstance(saved_result.get("retrieved_docids"), list)
                    and isinstance(saved_result.get("api_calls"), list)
                ):
                    print(
                        f"Query {qid} already has a complete atomic result; "
                        "skipping generation."
                    )
                    per_query_summaries.append(saved_summary)
                    per_query_results[qid] = saved_item
                    continue
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                print(
                    f"Existing query result is not reusable ({query_result_file}): "
                    f"{error}"
                )

        original_query = gt[qid]["query"]
        # Subqueries: use decomposed answers if available, else single query
        subqueries = [original_query]
        correct_answers = []
        try:
            correct_data = load_correct_answers(qid, data_dir)
            subqueries = correct_data.get("subqueries") or [original_query]
            correct_answers = [
                a.get("correct_answer", "") for a in correct_data.get("correct_answers", [])
            ]
        except FileNotFoundError:
            pass

        task_id = str(uuid.uuid4())
        env_config = {
            "task_id": task_id,
            "query_id": qid,
            "timeout": args.timeout,
            "original_query": original_query,
            "subqueries": subqueries,
            "correct_answers": correct_answers,
            "output_dir": str(output_dir),
            "memory_url": args.memory_url,
            "memory_timeout": args.memory_timeout,
            "memory_system_name": args.memory_system,
            "run_id": args.run_id,
            "config_digest": args.config_digest,
            "no_memory": args.no_memory,
            "step_memory": args.step_memory,
            "store_eval_in_memory": args.store_eval_in_memory,
            "script_path": script_path,
            "searcher_type": args.searcher_type,
            "index_path": args.index_path,
            "corpus_path": args.corpus_path,
            "gpu": args.gpu,
            "agent_model": args.model_name,
            "agent_api_key": args.api_key,
            "agent_base_url": args.base_url,
            "model_name": args.embedding_model,
            "judge_model": args.judge_model,
            "judge_api_key": args.judge_api_key,
            "judge_base_url": args.judge_base_url,
            "judge_protocol": args.judge_protocol,
            "defer_judge_if_unavailable": args.defer_judge_if_unavailable,
            "ground_truth_path": str(ground_truth_path),
            "provider": args.provider,
            "mcp_url": args.mcp_url,
            "mcp_name": args.mcp_name,
            "qrel_data": qrel_data,
            "max_iterations": args.max_iterations,
            "max_tokens": args.max_tokens,
            "retrieval_top_k": args.retrieval_top_k,
            "snippet_max_tokens": args.snippet_max_tokens,
            "max_search_calls": args.max_search_calls,
        }
        print(f"Using environment server at {args.env_server_url} (env_name=browsecomp-plus, query_id={qid}).")
        env_client = None
        out = None
        try:
            env_client = EnvironmentClient(
                task_id=task_id,
                timeout=args.timeout,
                env_name="browsecomp-plus",
                base_url=args.env_server_url,
                env_config=env_config,
            )
            env_client.reset()
            step_response = env_client.step({"command": "run_sequential"})
            result = step_response.get("observation", {})
            out = {
                "metadata": result.get("metadata", {}),
                "query_id": str(qid),
                "tool_call_counts": result.get("tool_call_counts"),
                "usage": result.get("usage", {}),
                "status": result.get("status", "completed"),
                "retrieved_docids": result.get("retrieved_docids", []),
                "result": result.get("trace", []),
                "api_calls": result.get("api_calls", []),
                "final_query": result.get("final_query"),
                "predicted_answer": result.get("predicted_answer"),
                "correct_answer": result.get("correct_answer"),
                "judgement": result.get("judgement"),
                "judge_status": result.get("judge_status"),
                "judge_model": result.get("judge_model", args.judge_model),
                "judge_protocol": result.get(
                    "judge_protocol", args.judge_protocol
                ),
                "judge_error": result.get("judge_error"),
                "recall": result.get("recall"),
                "failure": result.get("failure"),
            }
        except Exception as error:
            if not args.record_query_errors_as_failures:
                raise
            out = _build_terminal_failure_result(
                qid=str(qid),
                correct_answer=gt[qid].get("answer", ""),
                judge_model=args.judge_model,
                judge_protocol=args.judge_protocol,
                error=error,
            )
            print(
                "Recorded terminal query failure as an incorrect result: "
                f"query_id={qid} kind={out['failure']['kind']}"
            )
        finally:
            if env_client is not None:
                try:
                    env_client.close()
                except Exception as close_error:
                    print(
                        "Warning: environment close failed after query "
                        f"{qid}: {type(close_error).__name__}: {close_error}"
                    )

        summary = _summarize_single_result(out, qrel_data=qrel_data)
        per_query_summaries.append(summary)
        per_query_results[qid] = {"summary": summary, "result": out}
        write_json_atomic(
            query_result_file,
            {
                "query_ids": [qid],
                "summary": summary,
                "per_query": {qid: per_query_results[qid]},
            },
        )
        print(f"Atomic query result written to {query_result_file}")

    # Aggregate summaries across all valid query IDs
    total_processed = sum(s["processed_evaluations"] for s in per_query_summaries)
    total_skipped = sum(s["skipped_evaluations"] for s in per_query_summaries)
    total_failed = sum(s.get("failed_evaluations", 0) for s in per_query_summaries)
    total_evaluated = sum(s["evaluated_responses"] for s in per_query_summaries)

    # Overall accuracy: count queries whose judgement is correct; denominator is
    # the total number of valid input query IDs (even if a particular query had
    # no judgement and is effectively incorrect).
    correct_queries = 0
    for s in per_query_summaries:
        acc = s.get("accuracy")
        # For our per-query summaries, accuracy is either 1.0, 0.0, or None.
        if acc is not None and acc >= 0.5:
            correct_queries += 1
    overall_accuracy = None
    if valid_ids:
        overall_accuracy = correct_queries / len(valid_ids)

    # Overall recall: average over queries where recall is not None
    recall_sum = 0.0
    recall_count = 0
    for s in per_query_summaries:
        r = s.get("recall")
        if r is not None:
            recall_sum += r
            recall_count += 1
    overall_recall = None
    if recall_count > 0:
        overall_recall = recall_sum / recall_count

    # Average final-query tool calls per tool across all valid response slots.
    total_tool_calls = {}
    for s in per_query_summaries:
        n = s.get("evaluated_responses", 0)
        tools = s.get("average_tool_calls") or {}
        for tool, avg_calls in tools.items():
            total_tool_calls[tool] = total_tool_calls.get(tool, 0.0) + avg_calls * n
    overall_avg_tool_calls = {}
    if valid_ids:
        for tool, total_calls in total_tool_calls.items():
            overall_avg_tool_calls[tool] = total_calls / len(valid_ids)

    # Overall calibration error: average over evaluations where it is defined
    calib_sum = 0.0
    calib_n = 0
    for s in per_query_summaries:
        n = s.get("evaluated_responses", 0)
        ce = s.get("calibration_error")
        if ce is not None and n:
            calib_sum += ce * n
            calib_n += n
    overall_calib = None
    if calib_n > 0:
        overall_calib = calib_sum / calib_n

    overall_summary = {
        "processed_evaluations": total_processed,
        "skipped_evaluations": total_skipped,
        "failed_evaluations": total_failed,
        "evaluated_responses": total_evaluated,
        "accuracy": overall_accuracy,
        "recall": overall_recall,
        "average_tool_calls": overall_avg_tool_calls,
        "calibration_error": overall_calib,
    }

    # Print aggregated human-readable summary
    print(
        f"Processed {total_processed} evaluations "
        f"({total_skipped} skipped, {total_failed} failed)"
    )
    print(f"Evaluated {total_evaluated} responses:")

    if overall_accuracy is not None:
        print(f"Accuracy: {overall_accuracy * 100:.2f}%")
    else:
        print("Accuracy: N/A")

    if overall_recall is not None:
        print(f"Recall: {overall_recall * 100:.2f}%")
    else:
        print("Recall: N/A")

    if overall_avg_tool_calls:
        formatted_tools = {
            k: float(f"{v:.2f}") for k, v in overall_avg_tool_calls.items()
        }
        print(f"Average Tool Calls: {formatted_tools}")
    else:
        print("Average Tool Calls: {}")

    if overall_calib is not None:
        print(f"Calibration Error: {overall_calib * 100:.2f}%")
    else:
        print("Calibration Error: N/A")

    # Write a single JSON file summarizing all query IDs, plus per-query details
    if len(valid_ids) == 1:
        out_file = output_dir / f"query_{valid_ids[0]}_result.json"
    else:
        out_file = output_dir / "results_summary.json"
    write_json_atomic(
        out_file,
        {
            "query_ids": valid_ids,
            "summary": overall_summary,
            "per_query": per_query_results,
        },
    )
    print(f"Result summary written to {out_file}")


if __name__ == "__main__":
    main()
