#!/usr/bin/env python3
"""Unified command-line entry point for MemoryArena baselines and tasks."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parent
CODE_ROOT = PACKAGE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from Method_memoryarena.registry import (  # noqa: E402
    BACKENDS,
    BACKEND_KEYS,
    canonical_key,
)


TASK_RUNNERS = {
    "math": PACKAGE_ROOT / "runtime" / "run_math.py",
    "physics": PACKAGE_ROOT / "runtime" / "run_math.py",
    "search": PACKAGE_ROOT / "runtime" / "run_search.py",
    "travel": PACKAGE_ROOT / "runtime" / "run_travel.py",
    "shopping": PACKAGE_ROOT / "runtime" / "run_shopping.py",
}

TASK_ALIASES = {
    "math": "math",
    "formal_reasoning_math": "math",
    "phys": "physics",
    "physics": "physics",
    "formal_reasoning_phys": "physics",
    "search": "search",
    "progressive_search": "search",
    "travel": "travel",
    "group_travel_planner": "travel",
    "shopping": "shopping",
    "webshop": "shopping",
    "bundled_shopping": "shopping",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def load_config(path: Path) -> dict[str, Any]:
    config_path = path.expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"config file does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"config root must be a JSON object: {config_path}")
    return payload


def infer_task(config: Mapping[str, Any]) -> str:
    candidates = [config.get("task_name")]
    env = config.get("env")
    if isinstance(env, Mapping):
        candidates.extend((env.get("env_name"), env.get("task_name")))
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = str(candidate).strip().lower().replace("-", "_")
        task = TASK_ALIASES.get(normalized)
        if task:
            return task
    raise ValueError(
        "cannot infer task from config; pass --task with one of: "
        + ", ".join(TASK_RUNNERS)
    )


def resolve_method(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "all":
        return normalized
    try:
        return canonical_key(normalized)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc


def prepare_config(
    source: Mapping[str, Any],
    *,
    method: str,
    task: str,
    memory_url: str | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Return an isolated runner config for one method."""

    config = copy.deepcopy(dict(source))
    memory = config.setdefault("memory", {})
    if not isinstance(memory, dict):
        raise ValueError("config field 'memory' must be a JSON object")
    memory["memory_system_name"] = method
    if memory_url:
        endpoint = memory_url.rstrip("/")
        # Existing task runners use different historical names. Keep all three
        # synchronized so one unified option behaves identically for every task.
        memory["server_url"] = endpoint
        memory["memory_url"] = endpoint
        memory["base_url"] = endpoint

    if output_root is not None:
        method_output = output_root.expanduser().resolve() / task / method
        output = config.setdefault("output", {})
        if not isinstance(output, dict):
            raise ValueError("config field 'output' must be a JSON object")
        output["output_dir"] = str(method_output)
        if task == "travel":
            output["log_dir"] = str(method_output / "logs")
            output["global_csv"] = str(method_output / "global_eval.csv")

    return config


def runner_command(task: str, config_path: Path) -> list[str]:
    return [sys.executable, str(TASK_RUNNERS[task]), "--config", str(config_path)]


def _write_config(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def command_methods(_args: argparse.Namespace) -> int:
    print("key\tlabel")
    for spec in BACKENDS:
        print(f"{spec.key}\t{spec.label}")
    return 0


def command_serve(args: argparse.Namespace) -> int:
    os.environ["MEMORYARENA_RETRIEVAL_TOP_K"] = str(args.top_k)
    os.environ["MEMORYARENA_CONTEXT_CHAR_BUDGET"] = str(args.context_char_budget)
    import uvicorn

    from Method_memoryarena.server import app

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def command_run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        source = load_config(args.config)
        task = args.task or infer_task(source)
        method_arg = resolve_method(args.method)
    except ValueError as exc:
        parser.error(str(exc))

    if method_arg == "all" and args.output_root is None:
        parser.error("--output-root is required when --method all is used")

    methods: Sequence[str] = BACKEND_KEYS if method_arg == "all" else (method_arg,)
    failures: list[tuple[str, int]] = []

    with tempfile.TemporaryDirectory(prefix="memoryarena-unified-") as temp_dir:
        temp_root = Path(temp_dir)
        for method in methods:
            try:
                config = prepare_config(
                    source,
                    method=method,
                    task=task,
                    memory_url=args.memory_url,
                    output_root=args.output_root,
                )
            except ValueError as exc:
                parser.error(str(exc))
            generated_config = temp_root / f"{task}_{method}.json"
            _write_config(generated_config, config)
            command = runner_command(task, generated_config)
            output_dir = config.get("output", {}).get("output_dir", "<runner default>")
            print(f"\n=== MemoryArena: task={task} method={method} ===", flush=True)
            print(f"output: {output_dir}", flush=True)
            if args.dry_run:
                print("command: " + " ".join(command), flush=True)
                continue

            completed = subprocess.run(command, cwd=args.workdir, check=False)
            if completed.returncode:
                failures.append((method, completed.returncode))
                if args.fail_fast:
                    break

    if failures:
        summary = ", ".join(f"{method} (exit {code})" for method, code in failures)
        print(f"\nFailed method runs: {summary}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified entry point for the twelve MemoryArena baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    methods_parser = subparsers.add_parser("methods", help="list available methods")
    methods_parser.set_defaults(handler=command_methods)

    serve_parser = subparsers.add_parser("serve", help="start the unified memory server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=_positive_int, default=8000)
    serve_parser.add_argument("--top-k", type=_positive_int, default=10)
    serve_parser.add_argument(
        "--context-char-budget", type=_positive_int, default=8000
    )
    serve_parser.set_defaults(handler=command_serve)

    run_parser = subparsers.add_parser("run", help="run one task with one or all methods")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument(
        "--task",
        choices=tuple(TASK_RUNNERS),
        help="omit when task_name/env.env_name in the config identifies the task",
    )
    run_parser.add_argument(
        "--method",
        required=True,
        help="method key/alias, or 'all' to run all twelve methods",
    )
    run_parser.add_argument(
        "--memory-url", help="override the memory server URL in the config"
    )
    run_parser.add_argument(
        "--output-root",
        type=Path,
        help="write results below OUTPUT_ROOT/TASK/METHOD; required for method=all",
    )
    run_parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="working directory passed to the existing task runner",
    )
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(handler=command_run, command_parser=run_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return args.handler(args, args.command_parser)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
