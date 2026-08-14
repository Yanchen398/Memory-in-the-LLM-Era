from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sqlite3
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class CatalogLookup:
    category_by_asin: Any
    name_by_asin: Any


SQLITE_CATALOG_SCHEMA_VERSION = "webshop-catalog-v1"


class _SQLiteCatalogBackend:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._local = threading.local()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            uri = f"{self.path.as_uri()}?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
            connection.execute("PRAGMA query_only = ON")
            self._local.connection = connection
        return connection

    def get(self, column: str, asin: Any, default: Any = None) -> Any:
        if asin is None:
            return default
        if column not in {"category", "name"}:
            raise ValueError(f"unsupported catalog column: {column}")
        row = self._connection().execute(
            f"SELECT {column} FROM catalog WHERE asin = ?",
            (str(asin),),
        ).fetchone()
        if row is None or row[0] is None:
            return default
        return row[0]


class SQLiteValueLookup:
    def __init__(self, backend: _SQLiteCatalogBackend, column: str) -> None:
        self.backend = backend
        self.column = column

    def get(self, key: Any, default: Any = None) -> Any:
        return self.backend.get(self.column, key, default)


def _catalog_paths(
    catalog_dir: Path,
    catalog_files: Optional[Iterable[Path]],
) -> List[Path]:
    return (
        sorted(catalog_dir.glob("*.json"))
        if catalog_files is None
        else sorted(Path(path) for path in catalog_files)
    )


def _catalog_fingerprint(catalog_paths: Iterable[Path], namespace: str) -> str:
    hasher = hashlib.sha256(namespace.encode("utf-8"))
    for path in catalog_paths:
        try:
            stat = path.stat()
            meta = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        except FileNotFoundError:
            meta = f"{path.resolve()}:missing"
        hasher.update(meta.encode("utf-8"))
    return hasher.hexdigest()


def catalog_sqlite_cache_path(
    catalog_dir: Path,
    catalog_files: Optional[Iterable[Path]] = None,
) -> Path:
    catalog_paths = _catalog_paths(catalog_dir, catalog_files)
    cache_dir = os.getenv("WEBSHOP_CATALOG_SQLITE_CACHE_DIR")
    cache_root = Path(cache_dir) if cache_dir else catalog_dir / ".cache_sqlite"
    cache_key = _catalog_fingerprint(catalog_paths, SQLITE_CATALOG_SCHEMA_VERSION)
    return cache_root / f"catalog_{cache_key}.sqlite3"


def _sqlite_catalog_is_ready(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as connection:
            schema = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            rows = connection.execute("SELECT COUNT(*) FROM catalog").fetchone()
        return (
            schema is not None
            and schema[0] == SQLITE_CATALOG_SCHEMA_VERSION
            and rows is not None
            and int(rows[0]) > 0
        )
    except (OSError, sqlite3.Error):
        return False


def build_sqlite_catalog_cache(
    catalog_dir: Path,
    catalog_files: Optional[Iterable[Path]] = None,
) -> Path:
    import ijson

    catalog_paths = _catalog_paths(catalog_dir, catalog_files)
    if not catalog_paths:
        raise RuntimeError("no product catalog files selected")
    missing = [str(path) for path in catalog_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing product catalog files: {missing}")

    cache_path = catalog_sqlite_cache_path(catalog_dir, catalog_paths)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if _sqlite_catalog_is_ready(cache_path):
        return cache_path

    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    lock_acquired = False
    started = time.time()
    while time.time() - started < 3600:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            lock_acquired = True
            break
        except FileExistsError:
            if _sqlite_catalog_is_ready(cache_path):
                return cache_path
            time.sleep(1)
    if not lock_acquired:
        raise TimeoutError(f"timed out waiting for SQLite catalog cache: {cache_path}")

    temporary = cache_path.with_name(
        f".{cache_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        connection = sqlite3.connect(str(temporary))
        try:
            connection.executescript(
                """
                PRAGMA journal_mode = OFF;
                PRAGMA synchronous = OFF;
                PRAGMA temp_store = FILE;
                CREATE TABLE catalog (
                    asin TEXT PRIMARY KEY,
                    category TEXT,
                    name TEXT
                ) WITHOUT ROWID;
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                ) WITHOUT ROWID;
                """
            )
            upsert = """
                INSERT INTO catalog (asin, category, name) VALUES (?, ?, ?)
                ON CONFLICT(asin) DO UPDATE SET
                    category = CASE
                        WHEN catalog.category IS NULL THEN excluded.category
                        ELSE catalog.category
                    END,
                    name = CASE
                        WHEN catalog.name IS NULL THEN excluded.name
                        ELSE catalog.name
                    END
            """
            batch: List[Tuple[str, Optional[str], Optional[str]]] = []
            for path in catalog_paths:
                with path.open("rb") as handle:
                    for entry in ijson.items(handle, "item"):
                        if not isinstance(entry, dict):
                            continue
                        product_information = entry.get("product_information")
                        nested_asin = (
                            product_information.get("ASIN")
                            if isinstance(product_information, dict)
                            else None
                        )
                        raw_asin = entry.get("asin") or nested_asin
                        if not raw_asin:
                            continue
                        asin = str(raw_asin).strip().upper()
                        category = (
                            str(entry["product_category"])
                            if entry.get("product_category")
                            else None
                        )
                        name = str(entry["name"]) if entry.get("name") else None
                        batch.append((asin, category, name))
                        if len(batch) >= 5000:
                            connection.executemany(upsert, batch)
                            batch.clear()
                if batch:
                    connection.executemany(upsert, batch)
                    batch.clear()
                connection.commit()

            source_manifest = [
                {
                    "path": str(path.resolve()),
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in catalog_paths
            ]
            connection.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?)",
                (
                    ("schema_version", SQLITE_CATALOG_SCHEMA_VERSION),
                    ("source_manifest", json.dumps(source_manifest, sort_keys=True)),
                ),
            )
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            row_count = connection.execute("SELECT COUNT(*) FROM catalog").fetchone()
            if integrity != ("ok",) or row_count is None or int(row_count[0]) <= 0:
                raise RuntimeError(
                    f"invalid SQLite catalog cache: integrity={integrity}, rows={row_count}"
                )
        finally:
            connection.close()
        os.replace(temporary, cache_path)
        if not _sqlite_catalog_is_ready(cache_path):
            raise RuntimeError(f"SQLite catalog cache failed readiness: {cache_path}")
        return cache_path
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def load_sqlite_catalog(
    catalog_dir: Path,
    catalog_files: Optional[Iterable[Path]] = None,
) -> CatalogLookup:
    cache_path = build_sqlite_catalog_cache(catalog_dir, catalog_files)
    backend = _SQLiteCatalogBackend(cache_path)
    return CatalogLookup(
        category_by_asin=SQLiteValueLookup(backend, "category"),
        name_by_asin=SQLiteValueLookup(backend, "name"),
    )


def normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_for_match(text))


def load_catalog(
    catalog_dir: Path,
    catalog_files: Optional[Iterable[Path]] = None,
) -> CatalogLookup:
    if os.getenv("WEBSHOP_CATALOG_BACKEND", "pickle").strip().lower() == "sqlite":
        return load_sqlite_catalog(catalog_dir, catalog_files)

    category_by_asin: Dict[str, str] = {}
    name_by_asin: Dict[str, str] = {}

    catalog_paths = _catalog_paths(catalog_dir, catalog_files)

    cache_dir = os.getenv("WEBSHOP_CATALOG_CACHE_DIR")
    cache_root = Path(cache_dir) if cache_dir else catalog_dir / ".cache"
    cache_path: Optional[Path] = None
    lock_path: Optional[Path] = None
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        cache_root = None

    if cache_root:
        hasher = hashlib.sha256()
        for path in catalog_paths:
            try:
                stat = path.stat()
                meta = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
            except FileNotFoundError:
                meta = f"{path}:missing"
            hasher.update(meta.encode("utf-8"))
        cache_key = hasher.hexdigest()
        cache_path = cache_root / f"catalog_{cache_key}.pkl"
        lock_path = cache_root / f"catalog_{cache_key}.lock"
        if cache_path.exists():
            try:
                with cache_path.open("rb") as handle:
                    cached = pickle.load(handle)
                if isinstance(cached, CatalogLookup):
                    return cached
            except Exception:
                try:
                    cache_path.unlink()
                except Exception:
                    pass

    def build_catalog() -> CatalogLookup:
        for path in catalog_paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(data, list):
                continue
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                asin = entry.get("asin") or entry.get("product_information", {}).get("ASIN")
                if not asin:
                    continue
                asin = str(asin).strip().upper()
                if asin not in category_by_asin and entry.get("product_category"):
                    category_by_asin[asin] = str(entry["product_category"])
                if asin not in name_by_asin and entry.get("name"):
                    name_by_asin[asin] = str(entry["name"])
        return CatalogLookup(category_by_asin=category_by_asin, name_by_asin=name_by_asin)

    if cache_path and lock_path:
        lock_acquired = False
        start = time.time()
        while time.time() - start < 30:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                lock_acquired = True
                break
            except FileExistsError:
                if cache_path.exists():
                    try:
                        with cache_path.open("rb") as handle:
                            cached = pickle.load(handle)
                        if isinstance(cached, CatalogLookup):
                            return cached
                    except Exception:
                        pass
                time.sleep(0.1)
        if lock_acquired:
            try:
                catalog = build_catalog()
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=str(cache_path.parent),
                        prefix=cache_path.stem,
                        suffix=".tmp",
                        delete=False,
                    ) as tmp:
                        pickle.dump(catalog, tmp)
                        temp_path = tmp.name
                    os.replace(temp_path, cache_path)
                except Exception:
                    pass
                return catalog
            finally:
                try:
                    os.unlink(lock_path)
                except Exception:
                    pass

    return build_catalog()


def get_category_segments(category: Optional[str]) -> List[str]:
    if not category:
        return []
    return [segment.strip().lower() for segment in category.split("\u203a") if segment.strip()]


def category_similarity(
    target_category: Optional[str],
    purchased_category: Optional[str],
) -> float:
    target_segments = set(get_category_segments(target_category))
    purchased_segments = set(get_category_segments(purchased_category))
    if not target_segments:
        return 0.0
    return len(target_segments & purchased_segments) / len(target_segments)


def name_similarity(target_name: Optional[str], purchased_name: Optional[str]) -> float:
    if not target_name or not purchased_name:
        return 0.0
    target_tokens = set(tokenize(target_name))
    purchased_tokens = set(tokenize(purchased_name))
    if not target_tokens:
        return 0.0
    return len(target_tokens & purchased_tokens) / len(target_tokens)


def compute_r_type(name_score: float, category_score: float) -> Tuple[float, Dict[str, Any]]:
    match_condition = (category_score >= 0.5) or (name_score > 0.2)
    r_type = 1.0
    if not match_condition:
        r_type = 0.5
    if name_score < 0.1:
        r_type = 0.1
    if name_score == 0.0:
        r_type = 0.0
    return r_type, {
        "name_similarity": name_score,
        "category_similarity": category_score,
        "match_condition": "category_similarity>=0.5 or name_similarity>0.2",
        "low_name_threshold": 0.1,
        "zero_name_threshold": 0.0,
    }


def compute_attribute_matches(
    attributes: List[str],
    purchased_name: Optional[str],
) -> Tuple[int, List[str], List[str]]:
    if not attributes:
        return 0, [], []
    normalized_name = normalize_for_match(purchased_name or "")
    matched: List[str] = []
    missing: List[str] = []
    for attr in attributes:
        normalized_attr = normalize_for_match(attr)
        if normalized_attr and normalized_attr in normalized_name:
            matched.append(attr)
        else:
            missing.append(attr)
    return len(matched), matched, missing


def compute_price_reward(
    purchased_price: Optional[float],
    price_constraints: Dict[str, Any],
) -> Tuple[Optional[float], Dict[str, Any]]:
    price_upper = price_constraints.get("price_upper")
    price_lower = price_constraints.get("price_lower")
    if price_upper is None and price_lower is None:
        return None, {
            "price_upper": price_upper,
            "price_lower": price_lower,
            "purchased_price": purchased_price,
            "within_upper": None,
            "within_lower": None,
        }
    if purchased_price is None:
        return 0.0, {
            "price_upper": price_upper,
            "price_lower": price_lower,
            "purchased_price": purchased_price,
            "within_upper": False,
            "within_lower": False,
        }
    within_upper = True if price_upper is None else purchased_price <= price_upper
    within_lower = True if price_lower is None else purchased_price >= price_lower
    return 1.0 if (within_upper and within_lower) else 0.0, {
        "price_upper": price_upper,
        "price_lower": price_lower,
        "purchased_price": purchased_price,
        "within_upper": within_upper,
        "within_lower": within_lower,
    }


def extract_chain_id_from_source(source_path: Path) -> Optional[str]:
    parent_name = source_path.parent.name
    match = re.match(r"(.+?)_ins(?:_.*)?$", parent_name)
    if match:
        return match.group(1)
    return None


def slugify_catalog_segment(text: str) -> str:
    normalized = text.lower().replace("&", " ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def category_to_catalog_filename(category: str) -> Optional[str]:
    segments = [segment.strip() for segment in category.split("\u203a") if segment.strip()]
    if len(segments) < 2:
        return None
    top = slugify_catalog_segment(segments[0])
    sub = slugify_catalog_segment(segments[1])
    if not top or not sub:
        return None
    return f"{top}_{sub}.json"


def load_domain_categories(domain_data_path: Path) -> Dict[str, List[str]]:
    data = json.loads(domain_data_path.read_text(encoding="utf-8"))
    categories_by_chain: Dict[str, List[str]] = {}
    if not isinstance(data, list):
        return categories_by_chain
    for entry in data:
        if not isinstance(entry, dict):
            continue
        chain_id = entry.get("chain_id")
        if not chain_id:
            continue
        path = entry.get("path", [])
        if not isinstance(path, list):
            continue
        categories: List[str] = []
        for step in path:
            if not isinstance(step, dict):
                continue
            category = step.get("product_category")
            if category:
                categories.append(str(category))
        if categories:
            categories_by_chain[str(chain_id)] = categories
    return categories_by_chain


def select_catalog_files(
    product_catalog_dir: Path,
    domain_data_path: Path,
    source_files: List[str],
) -> Optional[List[Path]]:
    chain_ids = {
        chain_id
        for source_file in source_files
        for chain_id in [extract_chain_id_from_source(Path(source_file))]
        if chain_id
    }
    if not chain_ids or not domain_data_path.exists():
        return None
    categories_by_chain = load_domain_categories(domain_data_path)
    categories: List[str] = []
    for chain_id in chain_ids:
        categories.extend(categories_by_chain.get(chain_id, []))
    filenames = {
        filename
        for category in categories
        for filename in [category_to_catalog_filename(category)]
        if filename
    }
    matched_paths = [
        product_catalog_dir / filename
        for filename in sorted(filenames)
        if (product_catalog_dir / filename).exists()
    ]
    return matched_paths or None


def compute_reward_for_step(
    step_result: Dict[str, Any],
    gt_step: Dict[str, Any],
    catalog: CatalogLookup,
) -> Dict[str, Any]:
    purchased_asin = step_result.get("purchased_asin")
    target_asin = step_result.get("target_asin") or gt_step.get("target_asin")
    purchased_asin_upper = str(purchased_asin).upper() if purchased_asin else None
    target_asin_upper = str(target_asin).upper() if target_asin else None

    requirements = gt_step.get("requirements", {})
    attributes = requirements.get("attributes") or []
    price_constraints = requirements.get("price_constraints") or {}

    target_name = step_result.get("target_name") or catalog.name_by_asin.get(target_asin_upper)
    purchased_name = step_result.get("purchased_name") or (
        catalog.name_by_asin.get(purchased_asin_upper) if purchased_asin_upper else None
    )
    num_attr_matches, matched_attrs, missing_attrs = compute_attribute_matches(
        attributes,
        purchased_name,
    )
    r_attr = (num_attr_matches / len(attributes)) if attributes else None
    r_price, price_details = compute_price_reward(
        step_result.get("purchased_price"),
        price_constraints,
    )
    target_category = catalog.category_by_asin.get(target_asin_upper)
    purchased_category = catalog.category_by_asin.get(purchased_asin_upper)
    name_score = name_similarity(target_name, purchased_name)
    category_score = category_similarity(target_category, purchased_category)
    r_type, r_type_details = compute_r_type(name_score, category_score)

    base_numerator = num_attr_matches + (r_price if r_price is not None else 0.0)
    base_denominator = len(attributes) + (1 if r_price is not None else 0)
    base_reward = (base_numerator / base_denominator) if base_denominator else 0.0
    reward = base_reward * r_type
    forced_full = False
    reward_reason = "computed"

    if not purchased_asin_upper:
        reward = 0.0
        reward_reason = "no_purchase"
    elif purchased_asin_upper == target_asin_upper:
        reward = 1.0
        forced_full = True
        reward_reason = "asin_match"

    return {
        "step": step_result.get("step"),
        "target_asin": target_asin_upper,
        "target_name": target_name,
        "purchased_asin": purchased_asin_upper,
        "purchased_name": purchased_name,
        "purchased_price": step_result.get("purchased_price"),
        "reward": reward,
        "success": reward == 1.0,
        "components": {
            "attributes": attributes,
            "num_attr_matches": num_attr_matches,
            "attr_match_ratio": r_attr,
            "matched_attributes": matched_attrs,
            "missing_attributes": missing_attrs,
            "r_price": r_price,
            "price_details": price_details,
            "r_type": r_type,
            "r_type_details": r_type_details,
            "target_category": target_category,
            "purchased_category": purchased_category,
            "name_similarity": name_score,
            "category_similarity": category_score,
            "num_option_matches": 0,
            "option_ignored": True,
        },
        "calculation": {
            "base_numerator": base_numerator,
            "base_denominator": base_denominator,
            "base_reward": base_reward,
            "final_reward": reward,
            "forced_full_reward": forced_full,
            "reward_reason": reward_reason,
        },
    }
