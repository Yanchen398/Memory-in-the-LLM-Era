#!/usr/bin/env python3
"""Generate one controlled MemoryArena config matrix for all 12 baselines."""

import argparse
import json
import os
import uuid
from pathlib import Path


BASELINES = {
    "amem": "amem",
    "memorybank": "memorybank",
    "memgpt": "memgpt",
    "mem0": "mem0",
    "mem0g": "mem0g",
    "memochat": "memochat",
    "zep": "zep",
    "memtree": "memtree",
    "memoryos": "memoryos",
    "memos": "memos",
    "memgas": "memgas",
    "lightmem": "lightmem",
}


def _agent(args):
    return {
        "model_name": args.model,
        "temperature": 0.0,
        "max_tokens": args.max_output_tokens,
        "backend": "openai",
        "provider": "openai",
        "api_key": "EMPTY",
        "base_url": args.model_base_url,
    }


def _memory(args, adapter, *, search=False):
    key = "memory_url" if search else "server_url"
    return {
        "use_step_memory": False,
        "memory_system_name": adapter,
        key: args.memory_url,
        "base_url": args.memory_url,
        "timeout": args.timeout,
        "retrieval_top_k": args.retrieval_top_k,
        "context_char_budget": args.context_char_budget,
    }


def _reasoning(args, adapter, subset):
    config_name = f"formal_reasoning_{subset}"
    return {
        "task_name": "math" if subset == "math" else "phys",
        "description": f"Unified local {config_name} with {adapter}",
        "agent": _agent(args),
        "memory": {
            **_memory(args, adapter),
            "judge_result_in_memory": False,
        },
        "env": {
            "env_name": config_name,
            "base_url": args.env_url,
            "timeout": args.timeout,
            "env_config": _agent(args),
        },
        "task_specific": {
            "max_steps": 20,
            "auto_eval_after_run": False,
            "start_index": 0,
            "dataset": {
                "local_path": str(
                    args.math_data_path
                    if subset == "math"
                    else args.physics_data_path
                ),
            },
        },
        "output": {
            "output_dir": str(args.result_root / config_name / adapter),
            "save_trajectories": True,
            "save_metrics": True,
        },
        "benchmark_contract": _contract(args),
    }


def _search(args, adapter):
    agent = _agent(args)
    agent.update(
        {
            "embedding_model": args.search_embedding_model,
            "max_iterations": 9,
            "max_search_calls": 8,
            "retrieval_top_k": args.retrieval_top_k,
            "snippet_max_tokens": 64,
        }
    )
    return {
        "task_name": "search",
        "description": f"Unified local progressive search with {adapter}",
        "agent": agent,
        "memory": {**_memory(args, adapter, search=True), "no_memory": False},
        "env": {
            "env_server_url": args.env_url,
            "script_path": None,
            "searcher_type": "faiss",
            "gpu": "",
            "timeout": args.timeout,
            "index_path": args.search_index_path,
            "corpus_path": args.search_corpus_path,
            "mcp_url": None,
            "mcp_name": "retrieval-mcp-server",
        },
        "task_specific": {
            "query_ids": [],
            "store_eval_in_memory": False,
            "data_dir": args.search_data_dir,
            "qrel_evidence": "topics-qrels/qrel_evidence.txt",
            "judge_model": args.judge_model,
            "judge_protocol": args.judge_protocol,
            "judge_base_url": args.judge_base_url,
            "judge_api_key": args.judge_api_key,
            "defer_judge_if_unavailable": not bool(args.judge_base_url),
            "record_query_errors_as_failures": True,
        },
        "output": {"output_dir": str(args.result_root / "progressive_search" / adapter)},
        "benchmark_contract": _contract(args),
    }


def _travel(args, adapter):
    return {
        "task_name": "travel",
        "agent": _agent(args),
        "memory": _memory(args, adapter),
        "env": {
            "env_server_url": args.env_url,
            "env_config": {"judgement_mode": "hint"},
        },
        "task_specific": {"max_steps": 30, "start_index": 0},
        "output": {
            "output_dir": str(args.result_root / "group_travel_planner" / adapter),
            "log_dir": str(args.result_root / "group_travel_planner" / adapter / "logs"),
            "global_csv": str(args.result_root / "group_travel_planner" / adapter / "global_eval.csv"),
        },
        "benchmark_contract": _contract(args),
    }


def _shopping(args, adapter):
    return {
        "task_name": "shopping",
        "agent": _agent(args),
        "memory": _memory(args, adapter),
        "env": {
            "env_name": "webshop",
            "env_server_url": args.env_url,
            "timeout": args.timeout,
            "env_config": {
                "upstream_env_server_base": args.shopping_upstream_url,
                "bootstrap_upstream_env": True,
                "restart_upstream_env": False,
                "upstream_launch_module": "env.env_systems.web_shopping_env.runtime.service.launch_lite",
                "upstream_webshop_data_root": args.shopping_data_root,
                "upstream_limit_goals": -1,
                "upstream_ready_timeout": args.timeout,
                "reuse_env": True,
                "action_format": "react",
                "enable_feedback": False,
            },
        },
        "task_specific": {
            "task_category": "all",
            "task_file_limit": 0,
            "max_steps": 20,
            "split_steps": True,
            "resume": True,
            "include_history": False,
        },
        "output": {
            "output_dir": str(args.result_root / "bundled_shopping" / adapter),
            "save_trajectories": True,
            "save_metrics": True,
            "save_interactions": True,
        },
        "benchmark_contract": _contract(args),
    }


def _contract(args):
    return {
        "schema_version": "memoryarena.unified.v1",
        "run_id": args.run_id,
        "retrieval_top_k": args.retrieval_top_k,
        "context_char_budget": args.context_char_budget,
        "feedback_policy": args.feedback_policy,
        "max_model_len": args.max_model_len,
        "local_only": True,
        "answer_model_shared_across_methods": True,
    }


def build_matrix(args):
    builders = {
        "math": lambda adapter: _reasoning(args, adapter, "math"),
        "physics": lambda adapter: _reasoning(args, adapter, "phys"),
        "search": lambda adapter: _search(args, adapter),
        "travel": lambda adapter: _travel(args, adapter),
        "shopping": lambda adapter: _shopping(args, adapter),
    }
    matrix = {}
    for label, adapter in BASELINES.items():
        for task, builder in builders.items():
            matrix[(label, task)] = builder(adapter)
    return matrix


def _write_json_atomic(path: Path, payload, *, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing config: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--memory-url", required=True)
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--shopping-upstream-url", required=True)
    parser.add_argument("--shopping-data-root", required=True)
    parser.add_argument("--math-data-path", type=Path, required=True)
    parser.add_argument("--physics-data-path", type=Path, required=True)
    parser.add_argument("--search-data-dir", required=True)
    parser.add_argument("--search-index-path", required=True)
    parser.add_argument("--search-corpus-path", required=True)
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--search-embedding-model", default="/path/to/local/all-MiniLM-L6-v2")
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument("--judge-protocol", default="official_provider")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-api-key")
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--context-char-budget", type=int, default=8000)
    parser.add_argument(
        "--feedback-policy",
        choices=("none", "model_only", "gold_and_model"),
        default="model_only",
    )
    parser.add_argument("--timeout", type=int, default=3000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for name in ("max_model_len", "max_output_tokens", "retrieval_top_k", "context_char_budget", "timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    matrix = build_matrix(args)
    manifest = {
        "schema_version": "memoryarena.unified-config-matrix.v1",
        "baselines": BASELINES,
        "tasks": ["math", "physics", "search", "travel", "shopping"],
        "config_count": len(matrix),
    }
    targets = [
        args.output_dir / f"{task}_{label}.json"
        for label, task in matrix
    ] + [args.output_dir / "manifest.json"]
    if not args.overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to create a partial matrix because target files exist: "
                + ", ".join(str(path) for path in existing[:10])
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for (label, task), payload in matrix.items():
        path = args.output_dir / f"{task}_{label}.json"
        _write_json_atomic(path, payload, overwrite=args.overwrite)
    _write_json_atomic(
        args.output_dir / "manifest.json",
        manifest,
        overwrite=args.overwrite,
    )
    print(f"Wrote {len(matrix)} configs to {args.output_dir}")


if __name__ == "__main__":
    main()
