import os

from .simple_qa import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    simple_qa_session,
)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_PATH = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../Result/LOCOMO/amem/default/result.json")
)


def resolve_path(path_value, base_dir):
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def infer_modelname(model_name):
    if not model_name:
        return "amem"
    normalized = str(model_name).rstrip("/").rstrip("\\")
    inferred = os.path.basename(normalized)
    return inferred or "amem"


def run_amem(
    dataset_path,
    output_path=None,
    token_file=None,
    llm_model=DEFAULT_LLM_MODEL,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
    backend="openai",
    retrieve_k=10,
    ratio=1.0,
    start_idx=0,
    end_idx=None,
    config_path=None,
    modelname=None,
):
    base_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()

    dataset_path = resolve_path(dataset_path, base_dir)
    if not dataset_path:
        raise ValueError("AMEM requires 'dataset_path' in the config or CLI arguments.")

    output_path = resolve_path(output_path, base_dir) or DEFAULT_OUTPUT_PATH
    token_file = resolve_path(token_file, base_dir) or os.path.join(
        os.path.dirname(output_path), "token_tracker.json"
    )

    return simple_qa_session(
        dataset_path=dataset_path,
        model=llm_model or DEFAULT_LLM_MODEL,
        modelname=modelname or infer_modelname(llm_model),
        output_path=output_path,
        ratio=ratio if ratio is not None else 1.0,
        backend=backend or "openai",
        retrieve_k=retrieve_k if retrieve_k is not None else 10,
        start_idx=start_idx if start_idx is not None else 0,
        end_idx=end_idx,
        api_key=llm_api_key or DEFAULT_LLM_API_KEY,
        base_url=llm_base_url or DEFAULT_LLM_BASE_URL,
        embedding_model_name=embedding_model_name or DEFAULT_EMBEDDING_MODEL_NAME,
        token_file=token_file,
    )
