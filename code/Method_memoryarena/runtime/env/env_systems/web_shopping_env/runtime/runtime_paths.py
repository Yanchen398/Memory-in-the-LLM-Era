from __future__ import annotations

import os
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parent
MEMORYARENA_ROOT = RUNTIME_ROOT.parents[3]
WORKSPACE_ROOT = RUNTIME_ROOT.parents[5]


def _resolve_env_path(env_key: str, default: Path) -> Path:
    value = os.getenv(env_key)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve()


def _first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def default_webshop_data_root() -> Path:
    return _resolve_env_path(
        "MEMORYARENA_WEBSHOP_DATA_ROOT",
        MEMORYARENA_ROOT / "data" / "shopping",
    )


def _webshop_data_file(filename: str) -> Path:
    root = default_webshop_data_root()
    return _first_existing_path(root / filename, root / "data" / filename)


def default_items_file() -> Path:
    return _resolve_env_path(
        "MEMORYARENA_WEBSHOP_ITEMS_FILE",
        MEMORYARENA_ROOT / "data" / "shopping" / "items_shuffle.json",
    )


def default_attr_file() -> Path:
    return _resolve_env_path(
        "MEMORYARENA_WEBSHOP_ATTR_FILE",
        _first_existing_path(
            default_webshop_data_root() / "items_ins_v2.json",
            default_webshop_data_root() / "items_ins_v2_1000.json",
            default_webshop_data_root() / "data" / "items_ins_v2.json",
            default_webshop_data_root() / "data" / "items_ins_v2_1000.json",
        ),
    )


def default_human_attr_file() -> Path:
    return _resolve_env_path(
        "MEMORYARENA_WEBSHOP_HUMAN_ATTR_FILE",
        _webshop_data_file("items_human_ins.json"),
    )


def default_review_file() -> Path:
    return _resolve_env_path(
        "MEMORYARENA_WEBSHOP_REVIEW_FILE",
        _webshop_data_file("reviews.json"),
    )


def default_feat_conv_file() -> Path:
    return _resolve_env_path(
        "MEMORYARENA_WEBSHOP_FEAT_CONV_FILE",
        _webshop_data_file("feat_conv.pt"),
    )


def default_feat_ids_file() -> Path:
    return _resolve_env_path(
        "MEMORYARENA_WEBSHOP_FEAT_IDS_FILE",
        _webshop_data_file("feat_ids.pt"),
    )


def default_search_engine_root() -> Path:
    return _resolve_env_path(
        "MEMORYARENA_WEBSHOP_SEARCH_ROOT",
        MEMORYARENA_ROOT / "data" / "shopping" / "search_engine",
    )


def default_product_catalog_dir() -> Path:
    return _resolve_env_path(
        "MEMORYARENA_WEBSHOP_PRODUCT_CATALOG_DIR",
        MEMORYARENA_ROOT / "data" / "shopping" / "product_catalog",
    )


def default_domain_data_path() -> Path:
    return _resolve_env_path(
        "MEMORYARENA_WEBSHOP_DOMAIN_DATA_PATH",
        MEMORYARENA_ROOT / "data" / "shopping" / "domain_data.json",
    )
