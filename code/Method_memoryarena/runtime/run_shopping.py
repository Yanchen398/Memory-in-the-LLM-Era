from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback
import uuid
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_CODE_ROOT = Path(__file__).resolve().parents[2]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import tiktoken
except ImportError:
    tiktoken = None

from agent.webshop import LLMFatalError, WebShopAgent
from env.env_client import EnvironmentClient
from env.env_systems.web_shopping_env import (
    ensure_upstream_webshop_service,
    extract_port_from_base_url,
    render_port_template,
)
from env.env_systems.web_shopping_env.runtime import MEMORYARENA_ROOT, WORKSPACE_ROOT
from env.env_systems.web_shopping_env.runtime.controller.types import ActionFormat
from env.env_systems.web_shopping_env.runtime.runner import (
    build_instruction_for_step,
    build_single_step_task,
    collect_all_hf_category_prefixes,
    collect_task_files_from_hf,
    extract_product_name_from_text,
    hydrate_step_summary,
    index_conversation_by_asin,
    split_agent_instruction,
    summarize_all_steps_from_final_state,
    summarize_step_from_final_state,
    write_temp_task_file,
)
from env.env_systems.web_shopping_env.webshop_plus_client import WebshopAdapter
from Method_memoryarena.client import MemoryClient, configuration_digest

DEFAULT_CONFIG = MEMORYARENA_ROOT / "configs/web_shopping_configs/shopping_task.json"

logger = logging.getLogger("run_shopping")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write one terminal artifact without exposing a partial JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


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


def _setup_file_logger(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)

if tiktoken is not None:
    try:
        INTERACTION_HISTORY_ENCODING = tiktoken.encoding_for_model("gpt-4o-mini")
    except KeyError:
        INTERACTION_HISTORY_ENCODING = tiktoken.get_encoding("cl100k_base")
else:
    INTERACTION_HISTORY_ENCODING = None


def resolve_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, WORKSPACE_ROOT / path, MEMORYARENA_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (WORKSPACE_ROOT / path).resolve()


def _build_retry_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=0.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"POST"}),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_memory_system(
    memory_system_name: str,
    user_id: str,
    server_url: str,
    timeout: int = 300,
    run_id: Optional[str] = None,
    config_digest: Optional[str] = None,
) -> Optional[MemoryClient]:
    name = (memory_system_name or "none").strip().lower()
    if name in {"", "none", "nomem", "no"}:
        return None
    return MemoryClient(
        user_id=user_id,
        memory_system_name=name,
        base_url=server_url,
        session=_build_retry_session(),
        timeout=timeout,
        run_id=run_id,
        config_digest=config_digest,
    )


def collect_task_batches(args: argparse.Namespace) -> Dict[str, List[Path]]:
    """Collect task files grouped by category from HuggingFace.

    Returns an ordered ``{category_name: [Path, ...]}`` dict.

    * Single-file mode (``args.task_file`` resolves): one batch named after the file stem.
    * Otherwise: ``args.task_categories`` is a list of HF category prefixes
      (e.g. ``["baking", "beauty"]``); ``None`` loads every available category.
    """
    resolved_file = resolve_path(args.task_file)
    if resolved_file:
        return {resolved_file.stem: [resolved_file]}

    start = max(0, int(args.task_file_start or 0))
    limit = args.task_file_limit if args.task_file_limit > 0 else None
    categories = args.task_categories or collect_all_hf_category_prefixes()
    batches = {}
    for category in categories:
        all_files = collect_task_files_from_hf(category, None)
        end = start + limit if limit is not None else None
        selected = all_files[start:end]
        if not selected:
            raise ValueError(
                f"No Shopping task files selected for category={category!r}, "
                f"start={start}, limit={limit}, available={len(all_files)}"
            )
        batches[category] = selected
    return batches


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _workspace_relative_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    for root in (WORKSPACE_ROOT, MEMORYARENA_ROOT):
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    return str(path)


def _normalize_runner_metric(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _count_conversation_tokens(conversation: List[Dict[str, Any]]) -> Optional[int]:
    if INTERACTION_HISTORY_ENCODING is None:
        return None
    num_tokens = 0
    for message in conversation:
        num_tokens += 4
        num_tokens += len(INTERACTION_HISTORY_ENCODING.encode(message.get("role", "")))
        num_tokens += len(INTERACTION_HISTORY_ENCODING.encode(message.get("content", "")))
        reasoning_content = message.get("reasoning_content")
        if reasoning_content:
            num_tokens += len(INTERACTION_HISTORY_ENCODING.encode(reasoning_content))
    return num_tokens + 2


def load_config(config_path: Path) -> argparse.Namespace:
    cfg = _load_json(config_path)

    agent_cfg = cfg.get("agent", {})
    memory_cfg = cfg.get("memory", {})
    env_cfg = cfg.get("env", {})
    env_config = dict(env_cfg.get("env_config", {}))
    task_cfg = cfg.get("task_specific", {})
    output_cfg = cfg.get("output", {})
    benchmark_contract = cfg.get("benchmark_contract", {})

    api_key = agent_cfg.get("api_key") or os.getenv("OPENAI_API_KEY", "")
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)

    base_url = agent_cfg.get("base_url") or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
    if base_url:
        os.environ.setdefault("OPENAI_API_BASE", base_url)
        os.environ.setdefault("OPENAI_BASE_URL", base_url)

    # task_category: str | list[str] | "all"  →  list[str] | None (None means "all")
    raw_cats = task_cfg.get("task_category")
    if raw_cats is None:
        task_categories = None
    elif isinstance(raw_cats, str):
        task_categories = None if raw_cats == "all" else [raw_cats]
    else:
        flat = list(raw_cats)
        task_categories = None if flat == ["all"] else flat

    # Single-file shortcut (task_specific takes priority; env_config kept for compat)
    task_file = task_cfg.get("task_file") or env_config.get("task_file")

    # Human-readable run name used in log messages (not for output dirs)
    task_name = task_cfg.get("task_name") or cfg.get("task_name")
    if not task_name:
        if task_categories and len(task_categories) == 1:
            task_name = task_categories[0]
        elif task_file:
            task_name = Path(task_file).stem
        else:
            task_name = "shopping"

    return argparse.Namespace(
        task_name=task_name,
        model_name=agent_cfg.get("model_name", "gpt-5-mini"),
        temperature=agent_cfg.get("temperature", 0.0),
        max_tokens=agent_cfg.get("max_tokens", 512),
        context_window=agent_cfg.get("context_window"),
        api_key=api_key,
        model_base_url=base_url,
        backend=agent_cfg.get("backend", "openai"),
        memory_system=memory_cfg.get("memory_system_name", "none"),
        memory_server_url=memory_cfg.get("server_url") or memory_cfg.get("base_url") or "http://0.0.0.0:8000",
        memory_timeout=int(memory_cfg.get("timeout", 300)),
        use_step_memory=_as_bool(memory_cfg.get("use_step_memory"), False),
        run_id=benchmark_contract.get("run_id"),
        config_digest=configuration_digest(cfg),
        env_name=env_cfg.get("env_name", "webshop"),
        env_server_url=env_cfg.get("env_server_url") or env_cfg.get("base_url") or "http://0.0.0.0:8005",
        env_timeout=int(env_cfg.get("timeout", 300)),
        task_categories=task_categories,
        task_file=task_file,
        task_file_start=int(task_cfg.get("task_file_start", 0) or 0),
        task_file_limit=int(task_cfg.get("task_file_limit", env_config.get("task_file_limit", 1))),
        max_rounds=int(task_cfg.get("max_steps", 25)),
        reuse_env=_as_bool(env_config.get("reuse_env"), True),
        upstream_env_server_base=env_config.get("upstream_env_server_base") or os.getenv("ENV_SERVER_BASE") or "http://127.0.0.1:36004",
        env_id_cache_file=env_config.get("env_id_cache_file", ".env_id_cache_{port}_memoryarena_shopping.json"),
        action_format=env_config.get("action_format", "react"),
        enable_feedback=_as_bool(env_config.get("enable_feedback"), False),
        bootstrap_upstream_env=_as_bool(
            env_config.get("bootstrap_upstream_env", env_cfg.get("bootstrap_upstream_env")),
            True,
        ),
        restart_upstream_env=_as_bool(
            env_config.get("restart_upstream_env", env_cfg.get("restart_upstream_env")),
            False,
        ),
        upstream_limit_goals=int(
            env_config.get("upstream_limit_goals", env_cfg.get("upstream_limit_goals", -1))
        ),
        upstream_ready_timeout=float(
            env_config.get("upstream_ready_timeout", env_cfg.get("upstream_ready_timeout", 600))
        ),
        upstream_launch_module=env_config.get(
            "upstream_launch_module",
            env_cfg.get(
                "upstream_launch_module",
                "env.env_systems.web_shopping_env.runtime.service.launch_lite",
            ),
        ),
        upstream_python_executable=env_config.get(
            "upstream_python_executable",
            env_cfg.get("upstream_python_executable") or os.getenv("UPSTREAM_WEBSHOP_PYTHON"),
        ),
        upstream_webshop_data_root=env_config.get(
            "upstream_webshop_data_root",
            env_cfg.get("upstream_webshop_data_root") or "data/shopping",
        ),
        product_catalog_dir=env_config.get("product_catalog_dir"),
        domain_data_path=env_config.get("domain_data_path"),
        clear_env_id_cache_on_restart=_as_bool(
            env_config.get(
                "clear_env_id_cache_on_restart",
                env_cfg.get("clear_env_id_cache_on_restart"),
            ),
            True,
        ),
        clear_python_cache_on_restart=_as_bool(
            env_config.get(
                "clear_python_cache_on_restart",
                env_cfg.get("clear_python_cache_on_restart"),
            ),
            False,
        ),
        upstream_log_file=env_config.get("upstream_log_file", "logs/upstream_webshop_{port}.log"),
        split_steps=_as_bool(task_cfg.get("split_steps"), True),
        resume=_as_bool(task_cfg.get("resume"), True),
        include_history=_as_bool(task_cfg.get("include_history"), False),
        output_dir=output_cfg.get("output_dir", "results/shopping"),
        save_trajectories=_as_bool(output_cfg.get("save_trajectories"), True),
        save_metrics=_as_bool(output_cfg.get("save_metrics"), True),
        save_interactions=_as_bool(output_cfg.get("save_interactions"), True),
    )


def apply_overrides(args: argparse.Namespace, overrides: argparse.Namespace) -> argparse.Namespace:
    for key in (
        "memory_system",
        "memory_server_url",
        "task_file",
        "task_file_start",
        "task_file_limit",
        "max_rounds",
        "env_server_url",
        "upstream_env_server_base",
        "output_dir",
        "upstream_limit_goals",
        "upstream_ready_timeout",
        "upstream_launch_module",
        "upstream_python_executable",
        "upstream_webshop_data_root",
        "product_catalog_dir",
        "domain_data_path",
        "env_id_cache_file",
        "action_format",
        "upstream_log_file",
    ):
        value = getattr(overrides, key, None)
        if value is not None:
            setattr(args, key, value)
    # --task-category overrides task_categories (single value from CLI → one-element list)
    override_cat = getattr(overrides, "task_category", None)
    if override_cat is not None:
        args.task_categories = None if override_cat == "all" else [override_cat]
    for key in (
        "reuse_env",
        "enable_feedback",
        "clear_env_id_cache_on_restart",
        "clear_python_cache_on_restart",
    ):
        value = getattr(overrides, key, None)
        if value is not None:
            setattr(args, key, value)
    if getattr(overrides, "resume", None) is not None:
        args.resume = overrides.resume
    if getattr(overrides, "split_steps", None) is not None:
        args.split_steps = overrides.split_steps
    if getattr(overrides, "include_history", None) is not None:
        args.include_history = overrides.include_history
    if getattr(overrides, "bootstrap_upstream_env", None) is not None:
        args.bootstrap_upstream_env = overrides.bootstrap_upstream_env
    if getattr(overrides, "restart_upstream_env", None) is not None:
        args.restart_upstream_env = overrides.restart_upstream_env
    return args


def build_run_tag(args: argparse.Namespace) -> str:
    safe_model = args.model_name.rsplit("/", 1)[-1]
    memory_tag = (args.memory_system or "none").strip().lower()
    mode = "split" if args.split_steps else "full"
    return f"{safe_model}-{mode}-{memory_tag}"


def ensure_output_dirs(args: argparse.Namespace, task_scope: Optional[str] = None) -> Dict[str, Path]:
    p = Path(args.output_dir)
    output_root = p if p.is_absolute() else (MEMORYARENA_ROOT / p).resolve()
    if task_scope is None:
        task_scope = args.task_name
    artifact_root = output_root / task_scope / build_run_tag(args)
    step_result_dir = artifact_root / "step_results"
    interaction_dir = artifact_root / "interaction_history"
    tmp_task_dir = artifact_root / "tmp_tasks"
    log_dir = artifact_root / "logs"
    for path in (artifact_root, step_result_dir, interaction_dir, tmp_task_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    upstream_port = extract_port_from_base_url(args.upstream_env_server_base)
    cache_path = Path(render_port_template(str(args.env_id_cache_file), upstream_port))
    if not cache_path.is_absolute():
        cache_path = (artifact_root / cache_path).resolve()
    args.env_id_cache_file = str(cache_path)
    log_path = Path(render_port_template(str(args.upstream_log_file), upstream_port))
    if not log_path.is_absolute():
        log_path = (artifact_root / log_path).resolve()
    args.upstream_log_file = str(log_path)
    return {
        "artifact_root": artifact_root,
        "step_result_dir": step_result_dir,
        "interaction_dir": interaction_dir,
        "tmp_task_dir": tmp_task_dir,
        "log_dir": log_dir,
    }


def ensure_upstream_bootstrap(args: argparse.Namespace) -> None:
    if args.env_name != "webshop":
        return
    if not getattr(args, "bootstrap_upstream_env", False):
        return
    if getattr(args, "_upstream_bootstrap_done", False):
        return
    result = ensure_upstream_webshop_service(
        base_url=args.upstream_env_server_base,
        repo_root=MEMORYARENA_ROOT,
        reuse_env=args.reuse_env,
        cache_file=args.env_id_cache_file,
        restart=args.restart_upstream_env,
        python_executable=args.upstream_python_executable,
        launch_module=args.upstream_launch_module,
        limit_goals=args.upstream_limit_goals,
        ready_timeout_seconds=args.upstream_ready_timeout,
        clear_env_id_cache_on_restart=args.clear_env_id_cache_on_restart,
        clear_python_cache_on_restart=args.clear_python_cache_on_restart,
        log_file=args.upstream_log_file,
        env_overrides={
            "MEMORYARENA_WEBSHOP_DATA_ROOT": str(resolve_path(args.upstream_webshop_data_root) or args.upstream_webshop_data_root),
            "MEMORYARENA_WEBSHOP_ITEMS_FILE": str(MEMORYARENA_ROOT / "data" / "shopping" / "items_shuffle.json"),
            "MEMORYARENA_WEBSHOP_SEARCH_ROOT": str(MEMORYARENA_ROOT / "data" / "shopping" / "search_engine"),
        },
    )
    args._upstream_bootstrap_done = True
    args._upstream_bootstrap_result = result


def build_env_config(args: argparse.Namespace, task_file: Path) -> Dict[str, Any]:
    return {
        "task_file": str(task_file),
        "timeout": args.env_timeout,
        "reuse_env": args.reuse_env,
        "upstream_env_server_base": args.upstream_env_server_base,
        "env_id_cache_file": args.env_id_cache_file,
        "action_format": args.action_format,
        "max_rounds": args.max_rounds,
        "enable_feedback": args.enable_feedback,
        "bootstrap_upstream_env": args.bootstrap_upstream_env,
        "restart_upstream_env": args.restart_upstream_env,
        "upstream_limit_goals": args.upstream_limit_goals,
        "upstream_ready_timeout": args.upstream_ready_timeout,
        "upstream_launch_module": args.upstream_launch_module,
        "upstream_python_executable": args.upstream_python_executable,
        "upstream_webshop_data_root": args.upstream_webshop_data_root,
        "product_catalog_dir": args.product_catalog_dir,
        "domain_data_path": args.domain_data_path,
        "clear_env_id_cache_on_restart": args.clear_env_id_cache_on_restart,
        "clear_python_cache_on_restart": args.clear_python_cache_on_restart,
        "upstream_log_file": args.upstream_log_file,
    }


def task_memory_user_id(args: argparse.Namespace, task_def: Dict[str, Any], task_file: Path) -> str:
    task_key = task_def.get("task_id") or task_file.stem
    return f"shopping::{task_key}::{args.model_name}::{args.memory_system}"


def load_latest_task_results(result_json_dir: Path, task_file: Path) -> Optional[Dict[str, Any]]:
    pattern = f"{task_file.stem}_results_*.json"
    result_files = sorted(result_json_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not result_files:
        return None
    try:
        payload = _load_json(result_files[-1])
        payload.setdefault("result_file", str(result_files[-1]))
        return payload
    except Exception:
        return None


def load_latest_step_artifacts(interaction_dir: Path, task_file: Path, total_steps: int) -> Dict[int, Dict[str, Any]]:
    latest_files: Dict[int, Path] = {}
    prefix = f"eval_{task_file.stem}_step_"
    for json_path in interaction_dir.glob(f"{prefix}*.json"):
        name = json_path.stem
        try:
            remainder = name[len(prefix):]
            step_text = remainder.split("_", 1)[0]
            step_num = int(step_text)
        except Exception:
            continue
        if step_num > total_steps:
            continue
        current = latest_files.get(step_num)
        if current is None or json_path.stat().st_mtime > current.stat().st_mtime:
            latest_files[step_num] = json_path

    artifacts: Dict[int, Dict[str, Any]] = {}
    for step_num, json_path in latest_files.items():
        try:
            payload = _load_json(json_path)
        except Exception:
            continue
        artifact = extract_internal_interaction_artifact(payload)
        artifact["interaction_file"] = str(json_path)
        artifacts[step_num] = artifact
    return artifacts


def turns_to_conversation(turns: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    conversation: List[Dict[str, str]] = []
    if not turns:
        return conversation
    first_prompt = turns[0].get("prompt_source") or turns[0].get("prompt")
    if first_prompt:
        conversation.append({"role": "user", "content": first_prompt})
    for turn in turns:
        action = turn.get("action")
        if action:
            conversation.append({"role": "assistant", "content": action})
        observation = turn.get("observation") or {}
        state_text = observation.get("state") if isinstance(observation, dict) else str(observation)
        if state_text:
            conversation.append({"role": "user", "content": state_text})
    return conversation


def _normalize_api_message(
    role: str,
    content: Optional[str],
    reasoning_content: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "role": role,
        "content": content or "",
        "reasoning_content": reasoning_content,
    }


def _clone_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        _normalize_api_message(
            message.get("role", ""),
            message.get("content", ""),
            message.get("reasoning_content"),
        )
        for message in messages
    ]


def build_conversation_start_messages(action_format: str) -> List[Dict[str, Any]]:
    try:
        action_format_enum = ActionFormat(action_format)
    except ValueError:
        return []

    messages: List[Dict[str, Any]] = []
    for message in WebshopAdapter.conversation_start_dict.get(action_format_enum, ()):
        source = (message.get("from") or "").strip().lower()
        role = "assistant" if source in {"assistant", "gpt"} else "user"
        messages.append(_normalize_api_message(role, message.get("value", "")))
    return messages


def extract_internal_interaction_artifact(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Reconstruct a minimal artifact from the experiences structure
    experiences = payload.get("experiences") or []
    exp = experiences[0] if experiences else {}
    conversation = exp.get("conversation") or []
    final_state = ""
    for msg in reversed(conversation):
        if msg.get("role") == "user":
            final_state = msg.get("content", "")
            break
    return {
        "task_id": payload.get("task_id"),
        "task_file": payload.get("task_file"),
        "conversation": conversation,
        "turns": exp.get("turns") or [],
        "reward": exp.get("reward"),
        "final_state": final_state,
        "memory_entries": [],
    }


def _build_runner_conversation_from_turns(
    turns: List[Dict[str, Any]],
    fallback_conversation: List[Dict[str, Any]],
    final_observation: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not turns:
        return _clone_api_messages(fallback_conversation)

    first_input_messages = _clone_api_messages(turns[0].get("input_messages") or [])
    if first_input_messages:
        conversation = first_input_messages
    else:
        conversation = _clone_api_messages(fallback_conversation)

    for turn_idx, turn in enumerate(turns):
        output_message = turn.get("output_message") or {}
        output_content = output_message.get("content")
        if output_content is None and turn.get("raw_output") is not None:
            output_content = turn.get("raw_output")
        if output_content is not None:
            conversation.append(
                _normalize_api_message(
                    output_message.get("role", "assistant"),
                    output_content,
                    output_message.get("reasoning_content"),
                )
            )

        next_user_message: Optional[Dict[str, Any]] = None
        if turn_idx + 1 < len(turns):
            next_inputs = _clone_api_messages(turns[turn_idx + 1].get("input_messages") or [])
            if next_inputs:
                next_user_message = next_inputs[-1]
        else:
            observation = turn.get("observation") or final_observation or {}
            observation_text = (
                observation.get("state")
                if isinstance(observation, dict)
                else str(observation)
            )
            if observation_text:
                next_user_message = _normalize_api_message("user", observation_text)

        if next_user_message:
            conversation.append(next_user_message)

    return conversation


def _resolve_runner_score_and_success(artifact: Dict[str, Any]) -> Tuple[Any, Any]:
    info = artifact.get("info") or {}
    step_summary = artifact.get("step_summary")
    step_summaries = artifact.get("step_summaries") or []

    score = artifact.get("reward")
    success = None

    if isinstance(step_summary, dict):
        step_reward = step_summary.get("reward")
        if step_reward is not None:
            score = step_reward
        success = step_summary.get("match_ground_truth")

    if score is None and step_summaries:
        rewards = [item.get("reward") for item in step_summaries if item.get("reward") is not None]
        if rewards:
            normalized_rewards = []
            for reward in rewards:
                normalized = _normalize_runner_metric(reward)
                if isinstance(normalized, (int, float)):
                    normalized_rewards.append(float(normalized))
            if normalized_rewards:
                score = sum(normalized_rewards)

    if success is None and step_summaries:
        matches = [item.get("match_ground_truth") for item in step_summaries]
        if matches:
            success = all(bool(match) for match in matches)

    if success is None:
        success = info.get("match_ground_truth")
    if success is None:
        success = info.get("success")
    if success is None:
        success = bool(artifact.get("done")) and not artifact.get("error")

    return _normalize_runner_metric(score), _normalize_runner_metric(success)


def build_runner_compatible_interaction(
    artifact: Dict[str, Any],
    reward: Any,
) -> Dict[str, Any]:
    normalized = _build_runner_conversation_from_turns(
        artifact.get("turns") or [],
        artifact.get("conversation") or [],
        artifact.get("final_observation"),
    )
    turns: List[Dict[str, Any]] = []
    for message_idx, message in enumerate(normalized):
        if message.get("role") != "assistant":
            continue
        turn = {
            "assistant_message_index": message_idx,
            "input_messages": _clone_api_messages(normalized[:message_idx]),
            "output_message": _normalize_api_message(
                message.get("role", ""),
                message.get("content", ""),
                message.get("reasoning_content"),
            ),
        }
        input_tokens = _count_conversation_tokens(turn["input_messages"])
        if input_tokens is not None:
            turn["input_tokens"] = input_tokens
        turns.append(turn)
    return {
        "experience_index": 0,
        "reward": _normalize_runner_metric(reward),
        "conversation": normalized,
        "turns": turns,
    }


def build_interaction_history_payload(
    artifact: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    score, success = _resolve_runner_score_and_success(artifact)
    if score is None:
        score = 0.0
    if success is None:
        success = 0.0
    payload: Dict[str, Any] = {
        "task_id": artifact.get("task_id"),
        "task_file": _workspace_relative_path(artifact.get("env_task_file") or artifact.get("task_file")),
        "model": args.model_name,
        "endpoint": getattr(args, "backend", "openai"),
        "max_rounds": args.max_rounds,
        "max_tokens": args.max_tokens,
        "context_window": getattr(args, "context_window", None),
        "score": score,
        "success": success,
        "status": artifact.get("status", "failed" if artifact.get("error") else "completed"),
        "error": artifact.get("error"),
        "failure": artifact.get("failure"),
        "experiences": [build_runner_compatible_interaction(artifact, score)],
    }
    return payload


def summarize_step_artifact(
    step_def: Dict[str, Any],
    step_num: int,
    artifact: Dict[str, Any],
) -> Dict[str, Any]:
    stored_summary = artifact.get("step_summary")
    if isinstance(stored_summary, dict):
        summary = hydrate_step_summary(stored_summary, step_def, step_num)
    else:
        final_state = artifact.get("final_state") or ""
        summary = summarize_step_from_final_state(step_def, step_num, final_state)
        turns = artifact.get("turns") or []
        purchased_asin = summary.get("purchased_asin")
        if purchased_asin and not summary.get("purchased_name"):
            asin_lookup = index_conversation_by_asin(turns_to_conversation(turns))
            purchased_name = extract_product_name_from_text(
                purchased_asin,
                asin_lookup.get(purchased_asin, ""),
            )
            if purchased_name:
                summary["purchased_name"] = purchased_name
        summary = hydrate_step_summary(summary, step_def, step_num)
    interaction_file = artifact.get("interaction_file")
    if interaction_file:
        summary["log_file"] = interaction_file
    summary["episode_done"] = bool(artifact.get("done"))
    summary["error"] = artifact.get("error")
    summary["status"] = (
        "failed" if artifact.get("error") or not artifact.get("done") else "completed"
    )
    return summary


def build_resume_state(
    task_def: Dict[str, Any],
    task_file: Path,
    total_steps: int,
    dirs: Dict[str, Path],
) -> Tuple[Optional[int], Dict[int, str], List[Dict[str, Any]], Dict[int, Dict[str, Any]], Optional[Dict[str, Any]]]:
    feedback_map: Dict[int, str] = {}
    step_results: List[Dict[str, Any]] = []
    completed_steps: Dict[int, Dict[str, Any]] = {}
    step_artifacts = load_latest_step_artifacts(dirs["interaction_dir"], task_file, total_steps)
    existing_results = load_latest_task_results(dirs["step_result_dir"], task_file)

    if existing_results:
        enrich_task_result(existing_results)
        if existing_results.get("status") == "failed":
            return None, feedback_map, step_results, step_artifacts, existing_results

    if existing_results:
        for step_summary in existing_results.get("steps", []):
            step_num = step_summary.get("step")
            if not step_num or step_num > total_steps:
                continue
            summary = hydrate_step_summary(step_summary, task_def["steps"][step_num - 1], step_num)
            if step_num in step_artifacts and "log_file" not in summary:
                summary["log_file"] = step_artifacts[step_num].get("interaction_file")
            completed_steps[step_num] = summary

    if not completed_steps and step_artifacts:
        for step_num, artifact in step_artifacts.items():
            step_def = task_def["steps"][step_num - 1]
            completed_steps[step_num] = summarize_step_artifact(step_def, step_num, artifact)

    for step_num, summary in completed_steps.items():
        if step_num not in step_artifacts:
            step_artifacts[step_num] = {
                "step": step_num,
                "memory_entries": [
                    json.dumps(
                        {
                            "task": task_def.get("agent_instruction", ""),
                            "turn_idx": step_num,
                            "prompt_source": summary.get("target_description", ""),
                            "final_action": summary.get("feedback", ""),
                            "final_observation": {"state": summary.get("final_state", "")},
                            "reward": summary.get("reward"),
                            "done": True,
                            "info": {
                                "feedback": summary.get("feedback"),
                                "purchased_asin": summary.get("purchased_asin"),
                                "target_asin": summary.get("target_asin"),
                                "match_ground_truth": summary.get("match_ground_truth"),
                            },
                            "memory_mode": "resume_summary_backfill",
                        },
                        ensure_ascii=False,
                    )
                ],
            }

    next_step: Optional[int] = None
    for step_num in range(1, total_steps + 1):
        summary = completed_steps.get(step_num)
        if summary is None:
            next_step = step_num
            break
        feedback_map[step_num] = summary["feedback"]
        step_results.append(summary)

    return next_step, feedback_map, step_results, step_artifacts, existing_results


def build_memory_entries(agent: WebShopAgent, task_text: str, turns: List[Dict[str, Any]]) -> List[str]:
    entries: List[str] = []
    for turn in turns:
        obs = turn.get("observation") or {}
        obs_state = obs.get("state", "") if isinstance(obs, dict) else str(obs)
        entry = (
            f"Step {turn.get('turn_idx', '')}:\n"
            f"User Instruction: {turn.get('prompt_source', '')}\n"
            f"Agent Action: {turn.get('action', '')}\n"
            f"Env Observation: {obs_state}\n"
            f"Done: {turn.get('done')}"
        )
        entries.append(entry)
    return entries


def backfill_memory_from_artifacts(memory: Optional[MemoryClient], step_artifacts: Dict[int, Dict[str, Any]], max_step: int) -> None:
    if memory is None or max_step <= 0:
        return
    for step_num in sorted(step_artifacts):
        if step_num > max_step:
            continue
        artifact = step_artifacts[step_num]
        memory_entries = artifact.get("memory_entries") or []
        if not memory_entries:
            turns = artifact.get("turns") or []
            fallback_entries = []
            for i, turn in enumerate(turns):
                output_msg = turn.get("output_message") or {}
                agent_action = output_msg.get("content", "")
                input_msgs = turn.get("input_messages") or []
                user_msgs = [m for m in input_msgs if m.get("role") == "user"]
                user_instruction = user_msgs[-1].get("content", "") if user_msgs else ""
                if i + 1 < len(turns):
                    next_input = turns[i + 1].get("input_messages") or []
                    last_next = next_input[-1] if next_input else {}
                    env_observation = last_next.get("content", "") if last_next.get("role") == "user" else ""
                else:
                    env_observation = artifact.get("final_state", "")
                entry = (
                    f"Step {i + 1}:\n"
                    f"User Instruction: {user_instruction}\n"
                    f"Agent Action: {agent_action}\n"
                    f"Env Observation: {env_observation}\n"
                    f"Done: {i + 1 == len(turns)}"
                )
                fallback_entries.append(entry)
            memory_entries = fallback_entries
        for entry in memory_entries:
            memory.add(entry)


def persist_interaction_artifact(
    dirs: Dict[str, Path],
    source_task_file: Path,
    step_num: Optional[int],
    artifact: Dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    timestamp = artifact["timestamp"]
    if step_num is None:
        filename = f"eval_{source_task_file.stem}_{timestamp}.json"
    else:
        filename = f"eval_{source_task_file.stem}_step_{step_num}_{timestamp}.json"
    path = dirs["interaction_dir"] / filename
    write_json_atomic(path, build_interaction_history_payload(artifact, args))
    return path


def run_episode(
    args: argparse.Namespace,
    source_task_file: Path,
    env_task_file: Path,
    task_def: Dict[str, Any],
    task_instruction: str,
    memory: Optional[MemoryClient],
    step_num: Optional[int],
    use_step_memory: bool = False,
) -> Dict[str, Any]:
    task_id = str(uuid.uuid4())
    agent = WebShopAgent(
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
        base_url=args.model_base_url,
    )
    agent.reset()

    env_client: Optional[EnvironmentClient] = None
    observation: Dict[str, Any] = {}
    conversation: List[Dict[str, Any]] = build_conversation_start_messages(args.action_format)
    turns: List[Dict[str, Any]] = []
    last_result: Dict[str, Any] = {"observation": None, "reward": None, "done": False, "info": {}}
    error: Optional[str] = None
    start_time = time.time()
    memory_injected = False

    try:
        ensure_upstream_bootstrap(args)
        env_client = EnvironmentClient(
            task_id=task_id,
            env_name=args.env_name,
            base_url=args.env_server_url,
            timeout=args.env_timeout,
            env_config=build_env_config(args, env_task_file),
        )
        observation = env_client.reset()
        logger.info("env reset OK (step=%s)", step_num)
        for turn_idx in range(1, args.max_rounds + 1):
            prompt_source = observation.get("state") or json.dumps(observation, ensure_ascii=False)
            if memory is not None and not memory_injected:
                prompt = memory.wrap_user_prompt(prompt_source)
                memory_injected = True
            else:
                prompt = prompt_source

            input_messages = _clone_api_messages(conversation)
            input_messages.append(_normalize_api_message("user", prompt))

            logger.debug("turn %d: calling LLM (msgs=%d)", turn_idx, len(input_messages))
            action = agent.act_with_messages(input_messages)
            logger.debug("turn %d: action=%r", turn_idx, action[:120] if action else action)
            result = env_client.step(
                action,
                ground_truth=task_def.get("target_products"),
                need_judge=True,
            )
            observation = result["observation"]
            info = result.get("info") or {}
            raw_output = getattr(agent, "_last_raw_output", None)
            output_message = _normalize_api_message(
                "assistant",
                raw_output,
                getattr(agent, "_last_reasoning_content", None),
            )
            observation_message = _normalize_api_message(
                "user",
                observation.get("state") or json.dumps(observation, ensure_ascii=False),
            )
            conversation = input_messages + [output_message, observation_message]
            agent.record_turn(
                turn_idx=turn_idx,
                prompt=prompt,
                action=action,
                observation=observation,
                reward=result.get("reward"),
                done=result.get("done", False),
                info=info,
                raw_output=raw_output,
            )
            turn_record = {
                "turn_idx": turn_idx,
                "prompt_source": prompt_source,
                "action": action,
                "reward": result.get("reward"),
                "done": result.get("done", False),
                "info": info,
                "raw_output": raw_output,
                "assistant_message_index": len(input_messages),
                "input_messages": input_messages,
                "output_message": output_message,
            }
            if args.save_trajectories:
                turn_record["prompt"] = prompt
                turn_record["observation"] = observation
            else:
                turn_record["observation"] = {"state": observation.get("state", "")}
            turns.append(turn_record)
            # use_step_memory=True: write this turn's entry to memory immediately
            if use_step_memory and memory is not None:
                for entry in build_memory_entries(agent, task_instruction, [turn_record]):
                    memory.add(entry)
            last_result = result
            if result.get("done"):
                break
    except LLMFatalError:
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.error(
            "Exception in run_episode (step=%s, task=%s): %s\n%s",
            step_num,
            source_task_file.name,
            error,
            traceback.format_exc(),
        )
    finally:
        if env_client is not None:
            try:
                env_client.close()
            except Exception as exc:
                close_error = f"{type(exc).__name__}: {exc}"
                error = f"{error}; environment close failed: {close_error}" if error else close_error

    if error is None and not last_result.get("done", False):
        error = f"Episode did not reach done=True within {args.max_rounds} rounds"

    duration_seconds = round(time.time() - start_time, 2)
    final_observation = observation or {}
    final_state = ""
    if isinstance(final_observation, dict):
        final_state = final_observation.get("state", "")
    if not final_state and error:
        final_state = f"Error: {error}"

    memory_entries = build_memory_entries(agent, task_instruction, turns)
    # use_step_memory=False: write all turns at once when episode ends
    # use_step_memory=True: already written per-turn inside the loop
    if memory is not None and not use_step_memory and error is None:
        for entry in memory_entries:
            memory.add(entry)

    failure = None
    if error:
        failure = {
            "kind": _classify_failure(error),
            "stage": "episode",
            "message": error,
        }

    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "task_id": task_def.get("task_id"),
        "source_file": str(source_task_file),
        "env_task_file": str(env_task_file),
        "step": step_num,
        "task_instruction": task_instruction,
        "duration_seconds": duration_seconds,
        "done": last_result.get("done", False),
        "reward": last_result.get("reward"),
        "final_observation": final_observation,
        "final_state": final_state,
        "info": last_result.get("info") or {},
        "conversation": conversation,
        "turns": turns,
        "memory_entries": memory_entries,
        "error": error,
        "status": "failed" if error else "completed",
        "failure": failure,
    }


def enrich_task_result(result_payload: Dict[str, Any]) -> Dict[str, Any]:
    steps = result_payload.get("steps", [])
    total_steps = int(result_payload.get("total_steps", 0))
    matched_steps = sum(1 for step in steps if step.get("match_ground_truth"))
    purchased_steps = sum(1 for step in steps if step.get("purchased_asin"))
    step_failures = [
        {
            "step": step.get("step"),
            "message": step.get("error") or "episode did not reach done=True",
        }
        for step in steps
        if step.get("error") or step.get("status") == "failed" or step.get("episode_done") is False
    ]
    if total_steps <= 0:
        step_failures.append({"step": None, "message": "task contains no steps"})
    elif len(steps) != total_steps:
        step_failures.append({
            "step": None,
            "message": f"expected {total_steps} step results, got {len(steps)}",
        })
    expected_step_ids = list(range(1, total_steps + 1))
    actual_step_ids = [step.get("step") for step in steps]
    if total_steps > 0 and actual_step_ids != expected_step_ids:
        step_failures.append({
            "step": None,
            "message": f"unexpected step ordering: {actual_step_ids}",
        })
    task_error = result_payload.get("error")
    if task_error:
        step_failures.append({"step": None, "message": str(task_error)})
    result_payload["completed_steps"] = len(steps)
    result_payload["matched_steps"] = matched_steps
    result_payload["purchased_steps"] = purchased_steps
    result_payload["task_done"] = (
        len(steps) == total_steps and total_steps > 0 and not step_failures
    )
    result_payload["overall_success"] = (
        result_payload["task_done"] and matched_steps == total_steps
    )
    result_payload["status"] = "completed" if result_payload["task_done"] else "failed"
    metadata = result_payload.setdefault("metadata", {})
    metadata["terminal_failure"] = not result_payload["task_done"]
    if step_failures:
        result_payload["failure"] = {
            "kind": "episode_error",
            "stage": "task",
            "message": "; ".join(
                f"step {item['step']}: {item['message']}" for item in step_failures
            ),
            "episodes": step_failures,
            "failure_policy": "record_as_incorrect_and_continue",
        }
    else:
        result_payload["failure"] = None
    return result_payload


def _is_structurally_valid_task_result(result_payload: Dict[str, Any]) -> bool:
    total_steps = int(result_payload.get("total_steps", 0))
    steps = result_payload.get("steps")
    if result_payload.get("status") != "completed":
        return False
    if total_steps <= 0 or not isinstance(steps, list) or len(steps) != total_steps:
        return False
    expected_steps = list(range(1, total_steps + 1))
    actual_steps = [step.get("step") for step in steps]
    if actual_steps != expected_steps:
        return False
    return all(
        not step.get("error")
        and step.get("status", "completed") == "completed"
        and step.get("episode_done", True) is True
        for step in steps
    )


def _build_terminal_task_failure(
    task_file: Path,
    task_def: Dict[str, Any],
    error: Exception,
) -> Dict[str, Any]:
    message = _bounded_error_message(error)
    kind = _classify_failure(message)
    total_steps = len(task_def.get("steps", []))
    return {
        "metadata": {"terminal_failure": True},
        "task_id": task_def.get("task_id"),
        "source_file": str(task_file),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "total_steps": total_steps,
        "completed_steps": 0,
        "matched_steps": 0,
        "purchased_steps": 0,
        "task_done": False,
        "overall_success": False,
        "status": "failed",
        "steps": [],
        "failure": {
            "kind": kind,
            "stage": "task",
            "message": message,
            "failure_policy": "record_as_incorrect_and_continue",
        },
    }


def write_task_result(dirs: Dict[str, Path], task_file: Path, result_payload: Dict[str, Any]) -> Path:
    timestamp = result_payload.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = dirs["step_result_dir"] / f"{task_file.stem}_results_{timestamp}.json"
    write_json_atomic(path, result_payload)
    return path


def _process_task_file_split_impl(
    args: argparse.Namespace,
    task_file: Path,
    task_def: Dict[str, Any],
    dirs: Dict[str, Path],
    memory: Optional[MemoryClient],
) -> Dict[str, Any]:
    steps = task_def.get("steps", [])
    total_steps = len(steps)
    max_steps_allowed = task_def.get("global_constraints", {}).get("max_steps") or total_steps
    total_steps = min(total_steps, max_steps_allowed)
    prefix, sections = split_agent_instruction(task_def.get("agent_instruction", ""))

    feedback_map: Dict[int, str] = {}
    step_results: List[Dict[str, Any]] = []
    start_step = 1
    existing_results: Optional[Dict[str, Any]] = None

    if args.resume:
        next_step, feedback_map, step_results, step_artifacts, existing_results = build_resume_state(
            task_def,
            task_file,
            total_steps,
            dirs,
        )
        if next_step is None and existing_results is not None:
            return enrich_task_result(existing_results)
        if next_step is not None:
            start_step = next_step
        if start_step > 1 and memory is not None:
            backfill_memory_from_artifacts(memory, step_artifacts, start_step - 1)

    for idx in range(start_step - 1, total_steps):
        step_num = idx + 1
        logger.info("=== task=%s  step=%d/%d ===", task_file.name, step_num, total_steps)
        current_instruction = build_instruction_for_step(
            prefix,
            sections,
            step_num,
            feedback_map,
            include_history=args.include_history,
        )
        single_step_task = build_single_step_task(task_def, idx, current_instruction, total_steps=total_steps)
        step_task_path = write_temp_task_file(single_step_task, dirs["tmp_task_dir"])

        episode_artifact = run_episode(
            args=args,
            source_task_file=task_file,
            env_task_file=step_task_path,
            task_def=single_step_task,
            task_instruction=current_instruction,
            memory=memory,
            step_num=step_num,
            use_step_memory=args.use_step_memory,
        )

        interaction_path: Optional[Path] = None
        if args.save_interactions:
            interaction_path = persist_interaction_artifact(dirs, task_file, step_num, episode_artifact, args)
            episode_artifact["interaction_file"] = str(interaction_path)

        step_summary = summarize_step_artifact(single_step_task["steps"][0], step_num, episode_artifact)
        if interaction_path is not None:
            step_summary["log_file"] = str(interaction_path)
        episode_artifact["step_summary"] = step_summary
        if interaction_path is not None:
            write_json_atomic(
                interaction_path,
                build_interaction_history_payload(episode_artifact, args),
            )

        if episode_artifact.get("error"):
            logger.warning("step=%d finished with error: %s", step_num, episode_artifact["error"])
        else:
            logger.info(
                "step=%d done=%s reward=%s turns=%d",
                step_num,
                episode_artifact.get("done"),
                episode_artifact.get("reward"),
                len(episode_artifact.get("turns", [])),
            )
        feedback_map[step_num] = step_summary["feedback"]
        step_results.append(step_summary)

    result_payload = {
        "task_id": task_def.get("task_id"),
        "source_file": str(task_file),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_steps": total_steps,
        "split_steps": True,
        "resume": args.resume,
        "steps": step_results,
    }
    enrich_task_result(result_payload)
    result_path = write_task_result(dirs, task_file, result_payload)
    result_payload["result_file"] = str(result_path)
    return result_payload


def process_task_file_split(
    args: argparse.Namespace,
    task_file: Path,
    task_def: Dict[str, Any],
    dirs: Dict[str, Path],
) -> Dict[str, Any]:
    memory = None
    result_payload = None
    try:
        memory = get_memory_system(
            args.memory_system,
            task_memory_user_id(args, task_def, task_file),
            args.memory_server_url,
            args.memory_timeout,
            args.run_id,
            args.config_digest,
        )
        result_payload = _process_task_file_split_impl(
            args, task_file, task_def, dirs, memory
        )
        return result_payload
    finally:
        if memory is not None:
            memory.close(
                completed=(
                    result_payload is not None
                    and _is_structurally_valid_task_result(result_payload)
                )
            )


def _process_task_file_full_impl(
    args: argparse.Namespace,
    task_file: Path,
    task_def: Dict[str, Any],
    dirs: Dict[str, Path],
    memory: Optional[MemoryClient],
) -> Dict[str, Any]:
    existing_results = load_latest_task_results(dirs["step_result_dir"], task_file) if args.resume else None
    if existing_results is not None:
        return enrich_task_result(existing_results)

    total_steps = len(task_def.get("steps", []))
    max_steps_allowed = task_def.get("global_constraints", {}).get("max_steps") or total_steps
    total_steps = min(total_steps, max_steps_allowed)

    episode_artifact = run_episode(
        args=args,
        source_task_file=task_file,
        env_task_file=task_file,
        task_def=task_def,
        task_instruction=task_def.get("agent_instruction", ""),
        memory=memory,
        step_num=None,
        use_step_memory=args.use_step_memory,
    )
    interaction_path: Optional[Path] = None
    if args.save_interactions:
        interaction_path = persist_interaction_artifact(dirs, task_file, None, episode_artifact, args)
        episode_artifact["interaction_file"] = str(interaction_path)

    final_state = episode_artifact.get("final_state") or ""
    conversation = turns_to_conversation(episode_artifact.get("turns") or [])
    step_summaries = summarize_all_steps_from_final_state(
        task_def,
        final_state,
        conversation=conversation,
        total_steps=total_steps,
    )
    hydrated_steps = []
    for idx, step_summary in enumerate(step_summaries[:total_steps], start=1):
        hydrated = hydrate_step_summary(step_summary, task_def["steps"][idx - 1], idx)
        if interaction_path is not None:
            hydrated["log_file"] = str(interaction_path)
        hydrated_steps.append(hydrated)
    episode_artifact["step_summaries"] = hydrated_steps
    if interaction_path is not None:
        write_json_atomic(
            interaction_path,
            build_interaction_history_payload(episode_artifact, args),
        )

    result_payload = {
        "task_id": task_def.get("task_id"),
        "source_file": str(task_file),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_steps": total_steps,
        "split_steps": False,
        "resume": args.resume,
        "steps": hydrated_steps,
        "episode_done": bool(episode_artifact.get("done")),
        "error": episode_artifact.get("error"),
    }
    enrich_task_result(result_payload)
    result_path = write_task_result(dirs, task_file, result_payload)
    result_payload["result_file"] = str(result_path)
    return result_payload


def process_task_file_full(
    args: argparse.Namespace,
    task_file: Path,
    task_def: Dict[str, Any],
    dirs: Dict[str, Path],
) -> Dict[str, Any]:
    memory = None
    result_payload = None
    try:
        memory = get_memory_system(
            args.memory_system,
            task_memory_user_id(args, task_def, task_file),
            args.memory_server_url,
            args.memory_timeout,
            args.run_id,
            args.config_digest,
        )
        result_payload = _process_task_file_full_impl(
            args, task_file, task_def, dirs, memory
        )
        return result_payload
    finally:
        if memory is not None:
            memory.close(
                completed=(
                    result_payload is not None
                    and _is_structurally_valid_task_result(result_payload)
                )
            )


def process_task_file(
    args: argparse.Namespace,
    task_file: Path,
    dirs: Dict[str, Path],
) -> Dict[str, Any]:
    task_def = _load_json(task_file)
    steps = task_def.get("steps", [])
    if not steps:
        raise ValueError(f"No steps found in task definition: {task_file}")
    if args.split_steps:
        return process_task_file_split(args, task_file, task_def, dirs)
    return process_task_file_full(args, task_file, task_def, dirs)


def build_global_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "num_tasks": len(results),
        "num_task_done": sum(1 for item in results if item.get("task_done")),
        "num_task_success": sum(1 for item in results if item.get("overall_success")),
        "total_steps": sum(int(item.get("total_steps", 0)) for item in results),
        "matched_steps": sum(int(item.get("matched_steps", 0)) for item in results),
        "results": [
            {
                "task_id": item.get("task_id"),
                "source_file": item.get("source_file"),
                "task_done": item.get("task_done"),
                "overall_success": item.get("overall_success"),
                "status": item.get("status"),
                "failure": item.get("failure"),
                "completed_steps": item.get("completed_steps"),
                "matched_steps": item.get("matched_steps"),
                "total_steps": item.get("total_steps"),
                "result_file": item.get("result_file"),
            }
            for item in results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MemoryArena shopping task over upstream WebShop service.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--memory-system", type=str, default=None)
    parser.add_argument("--memory-server-url", type=str, default=None)
    parser.add_argument("--task-category", type=str, default=None,
                        help="HF category to run ('all' for every category, or e.g. 'baking')")
    parser.add_argument("--task-file", type=str, default=None)
    parser.add_argument("--task-file-start", type=int, default=None)
    parser.add_argument("--task-file-limit", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--env-server-url", type=str, default=None)
    parser.add_argument("--upstream-env-server-base", type=str, default=None)
    parser.add_argument("--upstream-limit-goals", type=int, default=None)
    parser.add_argument("--upstream-ready-timeout", type=float, default=None)
    parser.add_argument("--upstream-launch-module", type=str, default=None)
    parser.add_argument("--upstream-python-executable", type=str, default=None)
    parser.add_argument("--upstream-webshop-data-root", type=str, default=None)
    parser.add_argument("--product-catalog-dir", type=str, default=None)
    parser.add_argument("--domain-data-path", type=str, default=None)
    parser.add_argument("--env-id-cache-file", type=str, default=None)
    parser.add_argument("--action-format", type=str, default=None)
    parser.add_argument("--upstream-log-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--reuse-env", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-feedback", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--split-steps", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-history", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bootstrap-upstream-env", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--restart-upstream-env", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--clear-env-id-cache-on-restart", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--clear-python-cache-on-restart", action=argparse.BooleanOptionalAction, default=None)
    raw_args = parser.parse_args()

    args = load_config(resolve_path(raw_args.config) or Path(raw_args.config))
    args = apply_overrides(args, raw_args)

    task_batches = collect_task_batches(args)

    all_results: List[Dict[str, Any]] = []
    aggregate_failures: List[str] = []
    for batch_idx, (category, task_files) in enumerate(task_batches.items()):
        dirs = ensure_output_dirs(args, task_scope=category)
        if batch_idx == 0:
            run_log_path = dirs["log_dir"] / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            _setup_file_logger(run_log_path)
            logger.info("Log file: %s", run_log_path)
            logger.info(
                "Config: model=%s  memory=%s  output=%s",
                args.model_name, args.memory_system, args.output_dir,
            )
            ensure_upstream_bootstrap(args)

        logger.info("=== category: %s | %d task(s) ===", category, len(task_files))
        cat_results: List[Dict[str, Any]] = []
        for task_file in task_files:
            logger.info("--- processing task file: %s ---", task_file.name)
            try:
                result = process_task_file(args, task_file, dirs)
            except Exception as exc:
                logger.exception("Task failed while processing %s", task_file.name)
                try:
                    task_def = _load_json(task_file)
                except Exception:
                    task_def = {"task_id": task_file.stem, "steps": []}
                result = _build_terminal_task_failure(task_file, task_def, exc)
                result_path = write_task_result(dirs, task_file, result)
                result["result_file"] = str(result_path)
            logger.info("task %s done: overall_success=%s", task_file.name, result.get("overall_success"))
            cat_results.append(result)
            all_results.append(result)
            if result.get("status") == "failed":
                aggregate_failures.append(
                    f"{task_file.name}: "
                    f"{(result.get('failure') or {}).get('message', 'failed terminal state')}"
                )

        if args.save_metrics:
            summary_path = dirs["artifact_root"] / "summary.json"
            write_json_atomic(summary_path, build_global_summary(cat_results))

    # Cross-category summary when more than one batch was processed
    if args.save_metrics and len(task_batches) > 1:
        output_root = resolve_path(args.output_dir) or Path(args.output_dir)
        global_summary_path = output_root / build_run_tag(args) / "summary_all.json"
        global_summary_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(global_summary_path, build_global_summary(all_results))
        logger.info("Global summary written to %s", global_summary_path)

    if aggregate_failures:
        raise RuntimeError(
            f"{len(aggregate_failures)} shopping task(s) reached a failed terminal state: "
            + " | ".join(aggregate_failures)
        )


if __name__ == "__main__":
    main()
