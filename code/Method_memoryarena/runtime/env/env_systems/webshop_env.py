from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .base_env import BaseEnvironment
from .web_shopping_env.upstream_webshop import (
    ensure_upstream_webshop_service,
    extract_port_from_base_url,
    load_env_id,
    save_env_id,
)
from .web_shopping_env.runtime import MEMORYARENA_ROOT, WORKSPACE_ROOT
from .web_shopping_env.runtime.controller import BaseEnvClient
from .web_shopping_env.runtime.runtime_paths import default_domain_data_path, default_product_catalog_dir
from .web_shopping_env.webshop_plus_client import WebshopAdapter, WebshopPlusEnvClient


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class ReusableWebshopPlusEnvClient(WebshopPlusEnvClient):
    """Compatibility wrapper that can reuse an existing upstream env_id."""

    def __init__(
        self,
        env_server_base: str,
        task_file: str,
        data_len: int,
        *args,
        timeout: int = 300,
        env_id: Optional[int] = None,
        cache_file: str = ".env_id_cache_{port}_memoryarena_shopping.json",
        max_rounds: Optional[int] = None,
        product_catalog_dir: Optional[str] = None,
        domain_data_path: Optional[str] = None,
        enable_feedback: bool = False,
        **kwargs,
    ):
        cache_file = str(cache_file).replace(
            "{port}",
            str(extract_port_from_base_url(env_server_base)),
        )
        if env_id is None:
            super().__init__(
                env_server_base,
                task_file,
                data_len,
                *args,
                timeout=timeout,
                max_rounds=max_rounds,
                product_catalog_dir=product_catalog_dir,
                domain_data_path=domain_data_path,
                enable_feedback=enable_feedback,
                **kwargs,
            )
            save_env_id(self.env_id, cache_file)
            return

        BaseEnvClient.__init__(self, *args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        self.conversation_start = WebshopAdapter.conversation_start_dict[self.action_format]

        env_id_valid = False
        try:
            response = requests.get(
                f"{self.env_server_base}/observation?env_idx={env_id}",
                timeout=self.timeout,
            )
            env_id_valid = response.status_code == 200
        except Exception:
            env_id_valid = False

        if not env_id_valid:
            response = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
            response.raise_for_status()
            env_id = response.json()
            save_env_id(env_id, cache_file)

        self.env_id = env_id
        self.repo_root = WORKSPACE_ROOT
        self._init_task_state(task_file, max_rounds, enable_feedback, product_catalog_dir, domain_data_path)


class WebShopEnvironment(BaseEnvironment):
    """MemoryArena wrapper over the existing WebshopPlus evaluation client."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.config = config or {}
        self.timeout = int(self.config.get("timeout", 300))
        self.action_format = self.config.get("action_format", "react")
        self.reuse_env = _as_bool(self.config.get("reuse_env"), default=True)
        self.enable_feedback = _as_bool(self.config.get("enable_feedback"), default=False)
        self.max_rounds = self.config.get("max_rounds")
        self.upstream_env_server_base = self.config.get("upstream_env_server_base") or "http://127.0.0.1:36004"
        upstream_port = extract_port_from_base_url(self.upstream_env_server_base)
        self.env_id_cache_file = self.config.get(
            "env_id_cache_file",
            f".env_id_cache_{upstream_port}_memoryarena_shopping.json",
        )
        self.bootstrap_upstream_env = _as_bool(self.config.get("bootstrap_upstream_env"), default=False)
        self.restart_upstream_env = _as_bool(self.config.get("restart_upstream_env"), default=False)
        self.upstream_limit_goals = int(self.config.get("upstream_limit_goals", -1))
        self.upstream_ready_timeout = float(self.config.get("upstream_ready_timeout", 600))
        self.upstream_launch_module = self.config.get(
            "upstream_launch_module",
            "env.env_systems.web_shopping_env.runtime.service.launch_lite",
        )
        self.upstream_python_executable = self.config.get("upstream_python_executable")
        self.upstream_webshop_data_root = self.config.get("upstream_webshop_data_root") or "data/shopping"
        self.product_catalog_dir = self._resolve_repo_path(
            self.config.get("product_catalog_dir"),
            default_product_catalog_dir(),
        )
        self.domain_data_path = self._resolve_repo_path(
            self.config.get("domain_data_path"),
            default_domain_data_path(),
        )
        self.clear_env_id_cache_on_restart = _as_bool(
            self.config.get("clear_env_id_cache_on_restart"),
            default=True,
        )
        self.clear_python_cache_on_restart = _as_bool(
            self.config.get("clear_python_cache_on_restart"),
            default=False,
        )
        self.upstream_log_file = self.config.get("upstream_log_file")
        self.task_file = self._resolve_task_file(self.config.get("task_file"))
        self.task_def = self._load_task_definition(self.task_file)

        self.history: List[Dict[str, Any]] = []
        self.turn_count = 0
        self._done = False
        self._client: Optional[WebshopPlusEnvClient] = None
        self._upstream_bootstrap_done = False

        self._ensure_client()

    def _resolve_task_file(self, value: Any) -> Path:
        if not value:
            raise ValueError("WebShopEnvironment requires env_config.task_file")
        path = Path(str(value))
        if path.is_absolute():
            return path
        candidates = [Path.cwd() / path, WORKSPACE_ROOT / path, MEMORYARENA_ROOT / path]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return (WORKSPACE_ROOT / path).resolve()

    def _resolve_repo_path(self, value: Any, default: Path) -> Path:
        if not value:
            return default
        path = Path(str(value))
        if path.is_absolute():
            return path
        candidates = [Path.cwd() / path, WORKSPACE_ROOT / path, MEMORYARENA_ROOT / path]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return (WORKSPACE_ROOT / path).resolve()

    def _load_task_definition(self, task_file: Path) -> Dict[str, Any]:
        with open(task_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _ensure_upstream_service(self) -> None:
        if self._upstream_bootstrap_done or not self.bootstrap_upstream_env:
            return
        ensure_upstream_webshop_service(
            base_url=self.upstream_env_server_base,
            repo_root=MEMORYARENA_ROOT,
            reuse_env=self.reuse_env,
            cache_file=self.env_id_cache_file,
            restart=self.restart_upstream_env,
            python_executable=self.upstream_python_executable,
            launch_module=self.upstream_launch_module,
            limit_goals=self.upstream_limit_goals,
            ready_timeout_seconds=self.upstream_ready_timeout,
            clear_env_id_cache_on_restart=self.clear_env_id_cache_on_restart,
            clear_python_cache_on_restart=self.clear_python_cache_on_restart,
            log_file=self.upstream_log_file,
            env_overrides={
                "MEMORYARENA_WEBSHOP_DATA_ROOT": str(
                    self._resolve_repo_path(
                        self.upstream_webshop_data_root,
                        MEMORYARENA_ROOT / "data" / "shopping",
                    )
                ),
            },
        )
        self._upstream_bootstrap_done = True

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        self._ensure_upstream_service()
        env_id = load_env_id(self.env_id_cache_file) if self.reuse_env else None
        client_kwargs = {
            "env_server_base": self.upstream_env_server_base,
            "task_file": str(self.task_file),
            "data_len": 1,
            "timeout": self.timeout,
            "action_format": self.action_format,
            "max_rounds": self.max_rounds,
            "enable_feedback": self.enable_feedback,
            "product_catalog_dir": str(self.product_catalog_dir),
            "domain_data_path": str(self.domain_data_path),
        }
        if self.reuse_env:
            client_kwargs["env_id"] = env_id
            client_kwargs["cache_file"] = self.env_id_cache_file
            self._client = ReusableWebshopPlusEnvClient(**client_kwargs)
        else:
            self._client = WebshopPlusEnvClient(**client_kwargs)

    def _get_client(self) -> WebshopPlusEnvClient:
        self._ensure_client()
        assert self._client is not None
        return self._client

    def _current_purchases(self) -> List[Dict[str, Any]]:
        client = self._get_client()
        purchases = []
        for idx, asin in enumerate(getattr(client, "purchased_asins", []), start=1):
            price = None
            if idx - 1 < len(getattr(client, "purchased_prices", [])):
                price = client.purchased_prices[idx - 1]
            purchases.append({"idx": idx, "asin": asin, "price": price})
        return purchases

    def _get_upstream_state(self) -> Dict[str, Any]:
        client = self._get_client()
        try:
            return client._get("state")
        except Exception:
            return {}

    def _normalize_expected_asins(self, ground_truth: Any) -> List[str]:
        if ground_truth is None:
            ground_truth = self.task_def.get("target_products") or []
        if isinstance(ground_truth, dict):
            if "target_asins" in ground_truth:
                ground_truth = ground_truth["target_asins"]
            elif "target_products" in ground_truth:
                ground_truth = ground_truth["target_products"]
            elif "target_asin" in ground_truth:
                ground_truth = [ground_truth["target_asin"]]
            else:
                ground_truth = []
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        return [str(item).upper() for item in ground_truth or [] if item]

    def _build_judgement(self, ground_truth: Any) -> Dict[str, Any]:
        expected_asins = self._normalize_expected_asins(ground_truth)
        purchased_asins = [str(item).upper() for item in getattr(self._get_client(), "purchased_asins", [])]
        prefix_match = bool(expected_asins) and purchased_asins == expected_asins[: len(purchased_asins)]
        exact_match = bool(expected_asins) and purchased_asins == expected_asins
        return {
            "expected_asins": expected_asins,
            "purchased_asins": purchased_asins,
            "partial_match": prefix_match,
            "match_ground_truth": exact_match,
        }

    def _build_observation(self, state_text: str) -> Dict[str, Any]:
        upstream_state = self._get_upstream_state()
        observation = {
            "task_id": self.task_def.get("task_id"),
            "task_file": str(self.task_file),
            "task_description": self.task_def.get("task_description", ""),
            "instruction": self.task_def.get("agent_instruction", ""),
            "target_products": self.task_def.get("target_products", []),
            "turn_idx": self.turn_count,
            "done": self._done,
            "state": state_text,
            "purchases": self._current_purchases(),
            "upstream_env_id": getattr(self._get_client(), "env_id", None),
            "upstream_url": upstream_state.get("url"),
            "upstream_instruction_text": upstream_state.get("instruction_text"),
        }
        return observation

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        client = self._get_client()
        self.history = []
        self.turn_count = 0
        self._done = False
        client.reset(seed if seed is not None else 0)
        state_text = client.observe()
        self._current_observation = self._build_observation(state_text)
        return self._current_observation

    def step(
        self,
        action: Any,
        ground_truth: Any = None,
        need_judge: bool = False,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        client = self._get_client()
        step_output = client.step(str(action))
        self.turn_count += 1
        self._done = bool(step_output.done)

        info: Dict[str, Any] = {
            "task_id": self.task_def.get("task_id"),
            "task_file": str(self.task_file),
            "turn_idx": self.turn_count,
            "upstream_env_id": getattr(client, "env_id", None),
            "last_purchased_asin": getattr(client, "last_purchased_asin", None),
            "last_purchased_price": getattr(client, "last_purchased_price", None),
            "purchased_asins": list(getattr(client, "purchased_asins", [])),
            "purchased_prices": list(getattr(client, "purchased_prices", [])),
            "episode_done": getattr(client, "episode_done", self._done),
            "success": getattr(client, "success", False),
            "final_state": step_output.state if self._done else "",
            "soft_exit": "WEBSHOPPLUS SOFT EXIT" in step_output.state,
            "task_completed_summary": "WEBSHOPPLUS TASK COMPLETED" in step_output.state,
        }

        reward = None
        if need_judge:
            judgement = self._build_judgement(ground_truth)
            info.update(judgement)
            if self._done:
                reward = judgement["match_ground_truth"]

        observation = self._build_observation(step_output.state)
        self._current_observation = observation
        self.history.append(
            {
                "turn_idx": self.turn_count,
                "action": str(action),
                "done": self._done,
                "reward": reward,
                "info": info,
            }
        )
        return observation, reward, self._done, info

    def get_observation(self) -> Dict[str, Any]:
        return self._current_observation or self.reset()

    def close(self):
        if self._client is not None and not self.reuse_env:
            try:
                self._client.close()
            except Exception:
                pass
        self.history = []
        self.turn_count = 0
        self._done = False
        self._current_observation = None
        self._client = None
