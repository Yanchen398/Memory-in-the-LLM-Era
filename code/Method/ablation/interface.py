"""Unified command-line interface for all ablation experiment variants."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


GRANULARITIES = ("segment", "message")
STRUCTURES = ("tree", "graph")
_UNEXPANDED_ENV = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _unexpanded_variables(value: Any) -> set[str]:
    if isinstance(value, str):
        return {
            first or second
            for first, second in _UNEXPANDED_ENV.findall(value)
        }
    if isinstance(value, list):
        return set().union(*(_unexpanded_variables(item) for item in value), set())
    if isinstance(value, dict):
        return set().union(*(_unexpanded_variables(item) for item in value.values()), set())
    return set()


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"Config file does not exist: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    config = _expand_environment(config)
    missing = sorted(_unexpanded_variables(config))
    if missing:
        raise ValueError(
            "Set the environment variables referenced by the config: "
            + ", ".join(missing)
        )
    return config


def _format_paths(config: Mapping[str, Any], granularity: str, structure: str) -> dict[str, Any]:
    effective = dict(config)
    effective["memory_granularity"] = granularity
    effective["mid_term_structure"] = structure
    variables = {
        "memory_granularity": granularity,
        "mid_term_structure": structure,
    }
    for key in ("memory_path", "output_path"):
        if key not in effective:
            raise ValueError(f"Missing required config field: {key}")
        effective[key] = str(effective[key]).format(**variables)
    return effective


def _required(config: Mapping[str, Any], *names: str) -> None:
    missing = [name for name in names if config.get(name) in (None, "", [])]
    if missing:
        raise ValueError("Missing required config fields: " + ", ".join(missing))


def _run_core(config: Mapping[str, Any], granularity: str, structure: str, show: bool = False):
    import yaml

    effective = _format_paths(config, granularity, structure)
    _required(
        effective,
        "dataset_path",
        "output_path",
        "memory_path",
        "llm_model",
        "llm_base_url",
        "embedding_model_name",
    )
    if show:
        print(yaml.safe_dump(effective, allow_unicode=True, sort_keys=True))
        return None

    from .main_mp import run_sota

    with tempfile.TemporaryDirectory(prefix="ablation-config-") as temp_dir:
        config_path = Path(temp_dir) / "effective.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(effective, handle, allow_unicode=True, sort_keys=False)
        return run_sota(
            dataset_path=effective["dataset_path"],
            output_path=effective["output_path"],
            memory_path=effective["memory_path"],
            config_path=str(config_path),
            llm_model=effective["llm_model"],
            llm_api_key=effective.get("llm_api_key", "EMPTY"),
            llm_base_url=effective["llm_base_url"],
            embedding_model_name=effective["embedding_model_name"],
            memory_granularity=granularity,
            mid_term_structure=structure,
        )


def _command_run(args: argparse.Namespace) -> int:
    config = load_config(args.config_path)
    _run_core(config, args.memory_granularity, args.mid_term_structure, args.show_config)
    return 0


def _command_matrix(args: argparse.Namespace) -> int:
    config = load_config(args.config_path)
    for granularity in GRANULARITIES:
        for structure in STRUCTURES:
            print(f"\n=== Ablation: {granularity} + {structure} ===", flush=True)
            _run_core(config, granularity, structure, args.show_config)
    return 0


def _materialized_config(config: Mapping[str, Any], temp_dir: str) -> str:
    import yaml

    path = Path(temp_dir) / "effective.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, allow_unicode=True, sort_keys=False)
    return str(path)


def _command_graph_reretrieve(args: argparse.Namespace) -> int:
    config = load_config(args.config_path)
    from .graph_only_reretrieve import run_graph_only

    with tempfile.TemporaryDirectory(prefix="ablation-config-") as temp_dir:
        run_graph_only(
            config_path=_materialized_config(config, temp_dir),
            source_mode=args.source_mode,
            source_root=args.source_root,
            output_dir=args.output_dir,
            include_original_text=args.include_original_text,
            llm_base_url=args.llm_base_url,
            sample_concurrency=args.sample_concurrency,
            qa_concurrency=args.qa_concurrency,
            sample_max_retries=args.sample_max_retries,
            sample_indices=args.sample_indices,
            max_qa=args.max_qa,
        )
    return 0


def _command_tree_reretrieve(args: argparse.Namespace) -> int:
    config = load_config(args.config_path)
    from .tree_reretrieve import run_tree_reretrieve

    with tempfile.TemporaryDirectory(prefix="ablation-config-") as temp_dir:
        run_tree_reretrieve(
            config_path=_materialized_config(config, temp_dir),
            source_mode=args.source_mode,
            source_root=args.source_root,
            output_dir=args.output_dir,
            dialogue_top_k=args.dialogue_top_k,
            other_memory_top_k=args.other_memory_top_k,
            llm_base_url=args.llm_base_url,
            sample_concurrency=args.sample_concurrency,
            qa_concurrency=args.qa_concurrency,
            sample_max_retries=args.sample_max_retries,
            sample_indices=args.sample_indices,
            max_qa=args.max_qa,
        )
    return 0


def _command_memoryos_segment(args: argparse.Namespace) -> int:
    config = load_config(args.config_path)
    config["memory_granularity"] = "segment"
    _required(
        config,
        "dataset_path",
        "output_path",
        "memory_path",
        "llm_model",
        "llm_base_url",
        "embedding_model_name",
        "embedding_base_url",
    )
    from Method_memoryarena.memoryos.dispatch import run_memoryos_dispatch

    run_memoryos_dispatch(**config)
    return 0


def _add_retry_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-concurrency", type=int)
    parser.add_argument("--qa-concurrency", type=int)
    parser.add_argument("--sample-max-retries", type=int, default=2)
    parser.add_argument("--sample-indices")
    parser.add_argument("--max-qa", type=int)
    parser.add_argument("--llm-base-url")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified ablation experiment runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one granularity/structure combination")
    run_parser.add_argument("--config-path", required=True)
    run_parser.add_argument("--memory-granularity", choices=GRANULARITIES, required=True)
    run_parser.add_argument("--mid-term-structure", choices=STRUCTURES, required=True)
    run_parser.add_argument("--show-config", action="store_true")
    run_parser.set_defaults(handler=_command_run)

    matrix_parser = subparsers.add_parser("matrix", help="run all four core combinations")
    matrix_parser.add_argument("--config-path", required=True)
    matrix_parser.add_argument("--show-config", action="store_true")
    matrix_parser.set_defaults(handler=_command_matrix)

    graph_parser = subparsers.add_parser("graph-reretrieve", help="re-query persisted graph indexes")
    graph_parser.add_argument("--config-path", required=True)
    graph_parser.add_argument("--source-mode", choices=("segment_graph", "message_graph"), required=True)
    graph_parser.add_argument("--source-root")
    graph_parser.add_argument("--output-dir", required=True)
    graph_parser.add_argument("--include-original-text", action="store_true")
    _add_retry_options(graph_parser)
    graph_parser.set_defaults(handler=_command_graph_reretrieve)

    tree_parser = subparsers.add_parser("tree-reretrieve", help="re-query persisted tree indexes")
    tree_parser.add_argument("--config-path", required=True)
    tree_parser.add_argument("--source-mode", choices=("segment_tree", "message_tree"), required=True)
    tree_parser.add_argument("--source-root")
    tree_parser.add_argument("--output-dir", required=True)
    tree_parser.add_argument("--dialogue-top-k", type=int, required=True)
    tree_parser.add_argument("--other-memory-top-k", type=int, required=True)
    _add_retry_options(tree_parser)
    tree_parser.set_defaults(handler=_command_tree_reretrieve)

    memoryos_parser = subparsers.add_parser(
        "memoryos-segment", help="run the MemoryOS segment-granularity ablation"
    )
    memoryos_parser.add_argument("--config-path", required=True)
    memoryos_parser.set_defaults(handler=_command_memoryos_segment)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
