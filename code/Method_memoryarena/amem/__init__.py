"""A-MEM public API with optional runtime dependencies loaded lazily."""

from importlib import import_module

__version__ = "0.1.0"

_EXPORTS = {
    "run_amem": (".main", "run_amem"),
    "simple_qa_session": (".simple_qa", "simple_qa_session"),
    "SimpleMemAgent": (".simple_qa", "SimpleMemAgent"),
    "LLMController": (".memory_layer", "LLMController"),
    "AgenticMemorySystem": (".memory_layer", "AgenticMemorySystem"),
    "load_locomo_dataset": (".load_dataset", "load_locomo_dataset"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
