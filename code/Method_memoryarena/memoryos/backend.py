import os
import threading
import time
from pathlib import Path
from typing import Optional

try:
    from ..interface import (
        PreservingCloseState,
        close_resource,
        format_memory_prompt,
        normalize_chunk,
        resolve_context_char_budget,
        resolve_state_root,
        resolve_top_k,
        safe_user_id,
        select_endpoint,
    )
except ImportError:
    from Method_memoryarena.interface import (
        PreservingCloseState,
        close_resource,
        format_memory_prompt,
        normalize_chunk,
        resolve_context_char_budget,
        resolve_state_root,
        resolve_top_k,
        safe_user_id,
        select_endpoint,
    )


def _load_memoryos_class():
    try:
        from memoryos import Memoryos
        return Memoryos
    except ImportError:
        pass

    import importlib.util
    import site
    import sys

    candidate_roots = []
    try:
        candidate_roots.extend(site.getsitepackages())
    except Exception:
        pass
    user_site = site.getusersitepackages()
    if user_site:
        candidate_roots.append(user_site)

    for root in candidate_roots:
        package_dir = Path(root) / "memoryos"
        init_file = package_dir / "__init__.py"
        if not init_file.exists() or package_dir.resolve() == Path(__file__).resolve().parent:
            continue
        spec = importlib.util.spec_from_file_location(
            "_memoryos_vendor",
            init_file,
            submodule_search_locations=[str(package_dir)],
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            sys.modules["memoryos"] = module
            return module.Memoryos

    raise ImportError("Could not load pip package memoryos.Memoryos")


class MemoryOSMemorySystem:
    """MemoryArena memory-server adapter backed by the external MemoryOS library."""

    def __init__(self, user_id: Optional[str] = None, top_k: int = 10):
        from .main_lme_runner import _configure_library_runtime

        Memoryos = _load_memoryos_class()

        self.user_id = str(user_id or f"memoryarena_{time.time_ns()}")
        self.top_k = resolve_top_k(top_k, "MEMORYOS_RETRIEVAL_TOP_K")
        self.context_char_budget = resolve_context_char_budget(
            "MEMORYOS_CONTEXT_CHAR_BUDGET"
        )
        self.llm_model = os.getenv("MEMORYOS_LLM_MODEL") or os.getenv(
            "MEMORYARENA_LLM_MODEL", "Qwen3.5-9B"
        )
        self.llm_api_key = os.getenv("MEMORYOS_LLM_API_KEY") or os.getenv(
            "MEMORYARENA_LLM_API_KEY", "EMPTY"
        )
        self.llm_base_url = select_endpoint(
            self.user_id,
            kind="llm",
            method_list_env="MEMORYOS_LLM_BASE_URLS",
            method_single_env="MEMORYOS_LLM_BASE_URL",
        )
        embedding_model = os.getenv("MEMORYOS_EMBEDDING_MODEL") or os.getenv(
            "MEMORYARENA_EMBEDDING_MODEL", "/path/to/local/all-MiniLM-L6-v2"
        )
        embedding_api_key = os.getenv("MEMORYOS_EMBEDDING_API_KEY") or os.getenv(
            "MEMORYARENA_EMBEDDING_API_KEY", "EMPTY"
        )
        embedding_base_url = select_endpoint(
            self.user_id,
            kind="embedding",
            method_list_env="MEMORYOS_EMBEDDING_BASE_URLS",
            method_single_env="MEMORYOS_EMBEDDING_BASE_URL",
        )
        _configure_library_runtime(
            embedding_model,
            embedding_api_key,
            embedding_base_url,
            self.llm_model,
        )
        storage_root = resolve_state_root("memoryos", "MEMORYOS_DATA_ROOT")
        safe_id = safe_user_id(self.user_id)
        storage_path = storage_root / safe_id
        storage_path.mkdir(parents=True, exist_ok=True)
        self.memory = Memoryos(
            user_id=f"user_{safe_id}",
            assistant_id=f"assistant_{safe_id}",
            openai_api_key=self.llm_api_key,
            openai_base_url=self.llm_base_url,
            data_storage_path=str(storage_path),
            llm_model=self.llm_model,
            short_term_capacity=int(os.getenv("MEMORYOS_SHORT_TERM_CAPACITY", "7")),
            mid_term_heat_threshold=float(os.getenv("MEMORYOS_MID_TERM_HEAT_THRESHOLD", "5")),
            retrieval_queue_capacity=self.top_k,
        )
        self._lock = threading.RLock()
        self._close_state = PreservingCloseState()

    def add_chunk(self, chunk: str):
        text = normalize_chunk(chunk)
        if text is None:
            return {"stored": False, "skipped": True, "characters": 0}
        with self._lock:
            self._close_state.ensure_open(f"MemoryOS memory {self.user_id}")
            self.memory.add_memory(
                user_input="Completed MemoryArena subtask interaction",
                agent_response=text,
                meta_data={"source": "memoryarena_official_loop"},
            )
        return {"stored": True, "characters": len(text)}

    def wrap_user_prompt(self, prompt: str) -> str:
        with self._lock:
            self._close_state.ensure_open(f"MemoryOS memory {self.user_id}")
            result = self.memory.retriever.retrieve_context(
                user_query=str(prompt),
                user_id=self.memory.user_id,
            )
            entries = []
            for item in self.memory.short_term_memory.get_all():
                entries.append(
                    f"Input: {item.get('user_input', '')}\nOutput: {item.get('agent_response', '')}"
                )
            for page in result.get("retrieved_pages", []):
                entries.append(
                    f"Input: {page.get('user_input', '')}\nOutput: {page.get('agent_response', '')}"
                )
            for key in ("retrieved_user_knowledge", "retrieved_assistant_knowledge"):
                for item in result.get(key, []):
                    if item.get("knowledge"):
                        entries.append(str(item["knowledge"]))
        return format_memory_prompt(
            prompt,
            entries,
            char_budget=self.context_char_budget,
        )

    def _release_resources(self):
        seen = set()
        resources = close_resource(self.memory, "memoryos", seen=seen)
        if resources:
            return resources
        for component_name in (
            "retriever",
            "updater",
            "short_term_memory",
            "mid_term_memory",
            "long_term_memory",
        ):
            component = getattr(self.memory, component_name, None)
            component_resources = close_resource(
                component,
                f"memoryos.{component_name}",
                seen=seen,
            )
            resources.extend(component_resources)
            if not component_resources:
                resources.extend(
                    close_resource(
                        getattr(component, "client", None),
                        f"memoryos.{component_name}.client",
                        seen=seen,
                    )
                )
        resources.extend(
            close_resource(
                getattr(self.memory, "client", None),
                "memoryos.client",
                seen=seen,
            )
        )
        return resources

    def close(self, completed: bool = False):
        return self._close_state.close(completed, self._release_resources)


def create_backend(*, user_id=None, top_k=10):
    """Create MemoryOS through the common MemoryArena factory."""

    return MemoryOSMemorySystem(user_id=user_id, top_k=top_k)
