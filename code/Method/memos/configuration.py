import copy
import json
import os

import yaml


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_PATH = "/home/docker/IndepthMem/Dataset/LOCOMO/locomodemo.json"
DEFAULT_RESULT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../Result/LOCOMO/memos"))
DEFAULT_LLM_MODEL = "/home/docker/LLaMA-Factory/output/qwen2_5_lora_sft"
DEFAULT_LLM_API_KEY = "empty"
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_EMBEDDING_MODEL = "/home/docker/Model/all-mpnet-base-v2"


def deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_path(path_value, base_dir):
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def ensure_parent_dir(path):
    if path:
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)


def load_json_template(template_filename, custom_path=None, base_dir=None):
    if custom_path:
        template_path = resolve_path(custom_path, base_dir or CURRENT_DIR)
    else:
        template_path = os.path.join(CURRENT_DIR, "configs", template_filename)

    with open(template_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_openai_config(target_config, model_name=None, api_key=None, api_base=None):
    if model_name is not None:
        target_config["model_name_or_path"] = model_name
    if api_key is not None:
        target_config["api_key"] = api_key
    if api_base is not None:
        target_config["api_base"] = api_base


def load_runtime_config_file(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        if config_path.endswith(".json"):
            return json.load(f)
    raise ValueError(f"Unsupported config format: {config_path}")


def build_runtime_config(config=None, config_path=None):
    config = dict(config or {})
    base_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()

    version = config.get("version", "default")
    dataset_path = resolve_path(config.get("dataset_path", DEFAULT_DATASET_PATH), base_dir)
    result_dir = resolve_path(config.get("result_dir"), base_dir)
    if not result_dir:
        result_dir = os.path.join(DEFAULT_RESULT_ROOT, f"memos-{version}")
    result_dir = os.path.abspath(result_dir)

    runtime_config = {
        "version": version,
        "dataset_path": dataset_path,
        "num_workers": config.get("num_workers", 4),
        "top_k": config.get("top_k", 20),
        "ingestion_top_k": config.get("ingestion_top_k", 20),
        "result_dir": result_dir,
        "storage_dir": resolve_path(config.get("storage_dir"), base_dir) or os.path.join(result_dir, "storages"),
        "tmp_dir": resolve_path(config.get("tmp_dir"), base_dir) or os.path.join(result_dir, "tmp"),
        "search_results_path": resolve_path(config.get("search_results_path"), base_dir) or os.path.join(result_dir, "memos_locomo_search_results.json"),
        "response_results_path": resolve_path(config.get("response_results_path"), base_dir) or os.path.join(result_dir, "memos_locomo_responses.json"),
        "formatted_results_path": resolve_path(config.get("formatted_results_path"), base_dir) or os.path.join(result_dir, "result.json"),
        "token_file": resolve_path(config.get("token_file"), base_dir) or os.path.join(result_dir, "token_tracker.json"),
        "llm_model": config.get("llm_model", DEFAULT_LLM_MODEL),
        "llm_api_key": config.get("llm_api_key", DEFAULT_LLM_API_KEY),
        "llm_base_url": config.get("llm_base_url", DEFAULT_LLM_BASE_URL),
        "embedding_model_name": config.get("embedding_model_name", DEFAULT_EMBEDDING_MODEL),
        "response_model": config.get("response_model", config.get("llm_model", DEFAULT_LLM_MODEL)),
        "response_api_key": config.get("response_api_key", config.get("llm_api_key", DEFAULT_LLM_API_KEY)),
        "response_base_url": config.get("response_base_url", config.get("llm_base_url", DEFAULT_LLM_BASE_URL)),
        "mos_config_path": config.get("mos_config_path"),
        "mem_cube_config_path": config.get("mem_cube_config_path"),
        "mos_config_overrides": config.get("mos_config", {}),
        "mem_cube_config_overrides": config.get("mem_cube_config", {}),
        "max_turns_window": config.get("max_turns_window"),
        "enable_textual_memory": config.get("enable_textual_memory"),
        "enable_activation_memory": config.get("enable_activation_memory"),
        "enable_parametric_memory": config.get("enable_parametric_memory"),
        "graph_db_uri": config.get("graph_db_uri"),
        "graph_db_user": config.get("graph_db_user"),
        "graph_db_password": config.get("graph_db_password"),
        "graph_db_auto_create": config.get("graph_db_auto_create"),
    }

    runtime_config["mos_config_template"] = build_mos_config_template(runtime_config, base_dir)
    runtime_config["mem_cube_config_template"] = build_mem_cube_config_template(runtime_config, base_dir)
    return runtime_config


def build_mos_config_template(runtime_config, base_dir):
    mos_config = load_json_template(
        "mos_memos_config.json",
        custom_path=runtime_config.get("mos_config_path"),
        base_dir=base_dir,
    )

    mos_config["top_k"] = runtime_config["top_k"]
    apply_openai_config(
        mos_config["chat_model"]["config"],
        runtime_config["llm_model"],
        runtime_config["llm_api_key"],
        runtime_config["llm_base_url"],
    )
    apply_openai_config(
        mos_config["mem_reader"]["config"]["llm"]["config"],
        runtime_config["llm_model"],
        runtime_config["llm_api_key"],
        runtime_config["llm_base_url"],
    )
    mos_config["mem_reader"]["config"]["embedder"]["config"]["model_name_or_path"] = runtime_config["embedding_model_name"]

    for key in ("max_turns_window", "enable_textual_memory", "enable_activation_memory", "enable_parametric_memory"):
        if runtime_config.get(key) is not None:
            mos_config[key] = runtime_config[key]

    return deep_merge(mos_config, runtime_config.get("mos_config_overrides"))


def build_mem_cube_config_template(runtime_config, base_dir):
    mem_cube_config = load_json_template(
        "mem_cube_config.json",
        custom_path=runtime_config.get("mem_cube_config_path"),
        base_dir=base_dir,
    )

    apply_openai_config(
        mem_cube_config["text_mem"]["config"]["extractor_llm"]["config"],
        runtime_config["llm_model"],
        runtime_config["llm_api_key"],
        runtime_config["llm_base_url"],
    )
    apply_openai_config(
        mem_cube_config["text_mem"]["config"]["dispatcher_llm"]["config"],
        runtime_config["llm_model"],
        runtime_config["llm_api_key"],
        runtime_config["llm_base_url"],
    )
    mem_cube_config["text_mem"]["config"]["embedder"]["config"]["model_name_or_path"] = runtime_config["embedding_model_name"]

    graph_db_config = mem_cube_config["text_mem"]["config"]["graph_db"]["config"]
    if runtime_config.get("graph_db_uri") is not None:
        graph_db_config["uri"] = runtime_config["graph_db_uri"]
    if runtime_config.get("graph_db_user") is not None:
        graph_db_config["user"] = runtime_config["graph_db_user"]
    if runtime_config.get("graph_db_password") is not None:
        graph_db_config["password"] = runtime_config["graph_db_password"]
    if runtime_config.get("graph_db_auto_create") is not None:
        graph_db_config["auto_create"] = runtime_config["graph_db_auto_create"]

    return deep_merge(mem_cube_config, runtime_config.get("mem_cube_config_overrides"))


def build_mos_config(runtime_config, top_k=None):
    mos_config = copy.deepcopy(runtime_config["mos_config_template"])
    if top_k is not None:
        mos_config["top_k"] = top_k
    return mos_config


def build_mem_cube_config(runtime_config, user_id):
    mem_cube_config = copy.deepcopy(runtime_config["mem_cube_config_template"])
    mem_cube_config["user_id"] = user_id
    mem_cube_config["cube_id"] = user_id
    mem_cube_config["text_mem"]["config"]["graph_db"]["config"]["db_name"] = (
        f"{user_id.replace('_', '')}{runtime_config['version']}"
    )
    return mem_cube_config


def get_storage_path(runtime_config, user_id):
    return os.path.join(runtime_config["storage_dir"], user_id)


def get_tmp_search_results_path(runtime_config, group_idx):
    return os.path.join(runtime_config["tmp_dir"], f"memos_locomo_search_results_{group_idx}.json")
