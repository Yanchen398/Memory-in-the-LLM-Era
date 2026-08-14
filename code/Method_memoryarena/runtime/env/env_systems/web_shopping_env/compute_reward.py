#!/usr/bin/env python3
"""
Compute reward reports for MemoryArena WebShop step results.

Uses LLM-as-a-judge for attribute matching. Reward components:
  r_type  : name/category similarity to target product
  r_attr  : fraction of required attributes matched (via LLM or string)
  r_price : price-constraint satisfaction
  reward  = r_type * (r_attr + r_price) / (# terms with constraints)
  reward  = 1.0 if purchased ASIN == target ASIN (exact match)

Usage (run from MemoryArena root):
  # Single run directory:
  python env/env_systems/web_shopping_env/compute_reward.py \\
      --run-dir results/shopping/beauty/gpt-5-mini-split-long_context

  # All runs for one category:
  python env/env_systems/web_shopping_env/compute_reward.py --category beauty

  # All runs under results/shopping/:
  python env/env_systems/web_shopping_env/compute_reward.py --all

  # Skip already-computed; use --force to recompute:
  python env/env_systems/web_shopping_env/compute_reward.py --all --force

  # Disable LLM judge (string-matching only):
  python env/env_systems/web_shopping_env/compute_reward.py --all --no-llm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap: ensure MemoryArena root is importable regardless of cwd
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent  # .../env/env_systems/web_shopping_env/
_MEMORYARENA_ROOT = _HERE.parents[2]     # .../MemoryArena/
if str(_MEMORYARENA_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMORYARENA_ROOT))

from env.env_systems.web_shopping_env.runtime.reward_helpers import (  # noqa: E402
    CatalogLookup,
    compute_attribute_matches,
    compute_price_reward,
    compute_r_type,
    category_similarity,
    name_similarity,
    load_catalog,
    normalize_for_match,
    select_catalog_files,
)
from env.env_systems.web_shopping_env.runtime.runtime_paths import (  # noqa: E402
    MEMORYARENA_ROOT,
    default_domain_data_path,
    default_product_catalog_dir,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:  # type: ignore[misc]
        return iterable

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# LLM attribute judge
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = (
    "You are a strict product-attribute judge. "
    "Use only the product name. "
    "If an attribute is not explicitly stated, answer false. "
    "Return JSON only."
)

QWEN35_CHAT_EXTRA_BODY = {
    "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": False},
}


def _is_qwen_model(model: str | None) -> bool:
    return "qwen" in (model or "").lower()


@dataclass
class AttributeJudgeResult:
    matched_attributes: List[str]
    missing_attributes: List[str]
    used_llm: bool
    attempts: int
    raw_response: Optional[str]
    parsed_response: Optional[Dict[str, Any]]
    error: Optional[str]
    skipped: bool
    from_cache: bool


class AttributeJudge:
    def __init__(
        self,
        client: Any,
        model: str,
        max_retries: int = 3,
        retry_delay: float = 1.5,
        backend: str = "openai",
        endpoint: Optional[str] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backend = backend
        self.endpoint = endpoint
        self.records: List[Dict[str, Any]] = []
        self._cache: Dict[Tuple[str, Tuple[str, ...]], AttributeJudgeResult] = {}
        self.error_count = 0

    def judge(
        self,
        attributes: List[str],
        purchased_name: Optional[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> AttributeJudgeResult:
        context = context or {}

        if not attributes:
            result = AttributeJudgeResult(
                matched_attributes=[],
                missing_attributes=[],
                used_llm=False, attempts=0,
                raw_response=None, parsed_response=None,
                error=None, skipped=True, from_cache=False,
            )
            self._record(result, attributes, purchased_name, context)
            return result

        if not purchased_name:
            result = AttributeJudgeResult(
                matched_attributes=[],
                missing_attributes=list(attributes),
                used_llm=False, attempts=0,
                raw_response=None, parsed_response=None,
                error=None, skipped=True, from_cache=False,
            )
            self._record(result, attributes, purchased_name, context)
            return result

        cache_key = (purchased_name, tuple(attributes))
        cached = self._cache.get(cache_key)
        if cached is not None:
            hit = AttributeJudgeResult(
                matched_attributes=list(cached.matched_attributes),
                missing_attributes=list(cached.missing_attributes),
                used_llm=cached.used_llm, attempts=cached.attempts,
                raw_response=cached.raw_response,
                parsed_response=cached.parsed_response,
                error=cached.error, skipped=cached.skipped, from_cache=True,
            )
            self._record(hit, attributes, purchased_name, context)
            return hit

        prompt = (
            "Product name:\n"
            f"{purchased_name}\n\n"
            "Attributes:\n"
            + "\n".join(f"- {attr}" for attr in attributes)
            + '\n\nReturn JSON: {"matches":[{"attribute":"...","has_attribute":true/false}]}'
        )

        import re as _re

        def _parse_json(text: str) -> Dict[str, Any]:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = _re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
                cleaned = _re.sub(r"\n```$", "", cleaned).strip()
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start < 0 or end < 0 or end <= start:
                raise ValueError("No JSON object in response.")
            return json.loads(cleaned[start : end + 1])

        last_error: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                request = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                }
                if _is_qwen_model(self.model):
                    request["extra_body"] = QWEN35_CHAT_EXTRA_BODY
                response = self.client.chat.completions.create(
                    **request,
                )
                content = response.choices[0].message.content or ""
                parsed = _parse_json(content)
                matches = parsed.get("matches")
                if not isinstance(matches, list):
                    raise ValueError("Response JSON missing 'matches' list.")
                match_map: Dict[str, bool] = {
                    normalize_for_match(e["attribute"]): e["has_attribute"]
                    for e in matches
                    if isinstance(e, dict)
                    and isinstance(e.get("attribute"), str)
                    and isinstance(e.get("has_attribute"), bool)
                }
                matched = [a for a in attributes if match_map.get(normalize_for_match(a)) is True]
                missing = [a for a in attributes if a not in matched]
                result = AttributeJudgeResult(
                    matched_attributes=matched, missing_attributes=missing,
                    used_llm=True, attempts=attempt,
                    raw_response=content, parsed_response=parsed,
                    error=None, skipped=False, from_cache=False,
                )
                self._cache[cache_key] = result
                self._record(result, attributes, purchased_name, context)
                return result
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

        error_result = AttributeJudgeResult(
            matched_attributes=[], missing_attributes=list(attributes),
            used_llm=True, attempts=self.max_retries,
            raw_response=None, parsed_response=None,
            error=last_error or "LLM judge failed",
            skipped=False, from_cache=False,
        )
        self.error_count += 1
        print(
            f"[AttributeJudge] failed after {self.max_retries} attempts: {last_error}",
            file=sys.stderr,
        )
        self._record(error_result, attributes, purchased_name, context)
        return error_result

    def _record(
        self,
        result: AttributeJudgeResult,
        attributes: List[str],
        purchased_name: Optional[str],
        context: Dict[str, Any],
    ) -> None:
        self.records.append({
            "task_id": context.get("task_id"),
            "source_file": context.get("source_file"),
            "results_file": context.get("results_file"),
            "step": context.get("step"),
            "target_asin": context.get("target_asin"),
            "purchased_asin": context.get("purchased_asin"),
            "purchased_name": purchased_name,
            "attributes": list(attributes),
            "matched_attributes": list(result.matched_attributes),
            "missing_attributes": list(result.missing_attributes),
            "used_llm": result.used_llm,
            "attempts": result.attempts,
            "from_cache": result.from_cache,
            "skipped": result.skipped,
            "error": result.error,
            "raw_response": result.raw_response,
            "parsed_response": result.parsed_response,
        })

    def build_output(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "meta": {
                **meta,
                "backend": self.backend,
                "endpoint": self.endpoint,
                "model": self.model,
                "total_records": len(self.records),
                "error_count": self.error_count,
            },
            "records": self.records,
        }


# ---------------------------------------------------------------------------
# API key / judge factory
# ---------------------------------------------------------------------------

def _load_env_value(key: str, env_path: Path) -> Optional[str]:
    value = os.environ.get(key)
    if value:
        return value
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path)
        value = os.environ.get(key)
        if value:
            return value
    if not env_path.is_file():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if line.startswith(f"{key}="):
            _, value = line.split("=", 1)
            value = value.strip().strip("\"'")
            if value:
                os.environ[key] = value
                return value
    return None


def _find_env_file() -> Path:
    """Search for .env in MEMORYARENA_ROOT and parent directories."""
    for candidate in [
        MEMORYARENA_ROOT / ".env",
        MEMORYARENA_ROOT.parent / ".env",
        MEMORYARENA_ROOT.parent.parent / ".env",
        Path(".env").resolve(),
    ]:
        if candidate.is_file():
            return candidate
    return MEMORYARENA_ROOT / ".env"


def create_attribute_judge(
    env_path: Path,
    model: str,
    max_retries: int,
    retry_delay: float,
) -> AttributeJudge:
    if OpenAI is None:
        raise ImportError("openai package required for LLM judge. pip install openai")

    api_key = _load_env_value("OPENAI_API_KEY", env_path)
    base_url = _load_env_value("OPENAI_BASE_URL", env_path) or "https://api.openai.com/v1"
    if api_key:
        client = OpenAI(api_key=api_key, base_url=base_url)
        return AttributeJudge(
            client=client, model=model,
            max_retries=max_retries, retry_delay=retry_delay,
            backend="openai", endpoint=base_url,
        )

    azure_endpoint = _load_env_value("AZURE_OPENAI_ENDPOINT", env_path)
    azure_key = _load_env_value("AZURE_OPENAI_API_KEY", env_path)
    azure_deployment = _load_env_value("AZURE_DEPLOYMENT", env_path)
    azure_version = _load_env_value("AZURE_API_VERSION", env_path) or "2024-02-15-preview"
    if azure_endpoint and azure_key:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError("AzureOpenAI unavailable; upgrade the openai package.") from exc
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=azure_version,
        )
        return AttributeJudge(
            client=client, model=azure_deployment or model,
            max_retries=max_retries, retry_delay=retry_delay,
            backend="azure", endpoint=azure_endpoint,
        )

    raise RuntimeError(
        "No API key found. Set OPENAI_API_KEY (or Azure equivalents) in environment or .env file."
    )


# ---------------------------------------------------------------------------
# Reward computation (enhanced: LLM attribute judge)
# ---------------------------------------------------------------------------

def compute_reward_for_step(
    step_result: Dict[str, Any],
    gt_step: Dict[str, Any],
    catalog: CatalogLookup,
    attribute_judge: Optional[AttributeJudge] = None,
    judge_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute reward for a single step.

    When *attribute_judge* is provided the LLM evaluates attribute matches;
    otherwise falls back to string-matching (same as reward_helpers.py).
    """
    purchased_asin = step_result.get("purchased_asin")
    target_asin = step_result.get("target_asin") or gt_step.get("target_asin")
    purchased_asin_upper = str(purchased_asin).upper() if purchased_asin else None
    target_asin_upper = str(target_asin).upper() if target_asin else None

    requirements = gt_step.get("requirements", {})
    attributes: List[str] = requirements.get("attributes") or []
    price_constraints: Dict[str, Any] = requirements.get("price_constraints") or {}

    target_name = step_result.get("target_name") or catalog.name_by_asin.get(target_asin_upper)
    purchased_name = step_result.get("purchased_name") or (
        catalog.name_by_asin.get(purchased_asin_upper) if purchased_asin_upper else None
    )

    judge_details: Dict[str, Any] = {
        "used_llm": False, "attempts": 0,
        "error": None, "skipped": False, "from_cache": False,
    }
    if attribute_judge is not None:
        jr = attribute_judge.judge(attributes, purchased_name, context=judge_context or {})
        num_attr_matches = len(jr.matched_attributes)
        matched_attrs = list(jr.matched_attributes)
        missing_attrs = list(jr.missing_attributes)
        judge_details = {
            "used_llm": jr.used_llm, "attempts": jr.attempts,
            "error": jr.error, "skipped": jr.skipped, "from_cache": jr.from_cache,
        }
    else:
        num_attr_matches, matched_attrs, missing_attrs = compute_attribute_matches(
            attributes, purchased_name
        )

    r_attr = (num_attr_matches / len(attributes)) if attributes else None
    r_price, price_details = compute_price_reward(
        step_result.get("purchased_price"), price_constraints
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
            "attribute_judge": judge_details,
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


def _update_step_stats(
    stats: Dict[int, Dict[str, Any]], step_num: int, step_entry: Dict[str, Any]
) -> None:
    reward = step_entry["reward"]
    purchased_asin = step_entry.get("purchased_asin")
    if step_num not in stats:
        stats[step_num] = {
            "count": 0, "reward_sum": 0.0,
            "success_count": 0, "no_purchase_count": 0,
            "partial_reward_count": 0, "other_failure_count": 0,
            "rewards": [],
        }
    e = stats[step_num]
    e["count"] += 1
    e["reward_sum"] += reward
    e["rewards"].append(None if purchased_asin is None else reward)
    if reward == 1.0:
        e["success_count"] += 1
    elif purchased_asin is None:
        e["no_purchase_count"] += 1
    elif 0.0 < reward < 1.0:
        e["partial_reward_count"] += 1
    else:
        e["other_failure_count"] += 1


def _load_ground_truth(
    source_file: str, result_path: Path
) -> Optional[Tuple[Path, Dict[int, Dict[str, Any]]]]:
    """Resolve the ground-truth JSON for a result file.

    Try, in order:
    1. source_file as absolute path.
    2. MEMORYARENA_ROOT / source_file (for relative paths stored in results).
    3. hf_task_cache/[category]/[basename] (inferred from source_file parent name).
    """
    import re as _re

    m = _re.match(r"(item_\d+)_results", result_path.stem)
    basename = f"{m.group(1)}.json" if m else Path(source_file).name
    sf_path = Path(source_file)

    candidates: List[Path] = []
    if sf_path.is_absolute():
        candidates.append(sf_path)
    else:
        candidates.append(MEMORYARENA_ROOT / sf_path)
    candidates.append(MEMORYARENA_ROOT / "hf_task_cache" / sf_path.parent.name / basename)
    candidates.append(MEMORYARENA_ROOT / "hf_task_cache" / basename)

    for candidate in candidates:
        if candidate.is_file():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            steps_by_num: Dict[int, Dict[str, Any]] = {
                step["step"]: step
                for step in data.get("steps", [])
                if isinstance(step.get("step"), int)
            }
            return candidate, steps_by_num
    return None


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def collect_run_dirs(results_base: Path, category: Optional[str] = None) -> List[Path]:
    """Return all run directories (those containing step_results/).

    Layout: results_base/[category]/[run_tag]/step_results/
    """
    run_dirs: List[Path] = []
    search_roots = (
        [results_base / category] if category else
        [d for d in sorted(results_base.iterdir()) if d.is_dir()]
        if results_base.is_dir() else []
    )
    for root in search_roots:
        if not root.is_dir():
            continue
        for run_dir in sorted(root.iterdir()):
            if run_dir.is_dir() and (run_dir / "step_results").is_dir():
                run_dirs.append(run_dir)
    return run_dirs


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(MEMORYARENA_ROOT)
    except ValueError:
        return path


# ---------------------------------------------------------------------------
# Single-run processing
# ---------------------------------------------------------------------------

def process_run_dir(
    run_dir: Path,
    catalog: CatalogLookup,
    attribute_judge: Optional[AttributeJudge],
    output_name: str = "reward_report.json",
    force: bool = False,
) -> Dict[str, Any]:
    """Compute and save reward_report.json for one run directory."""
    output_path = run_dir / output_name
    rel = display_path(run_dir)

    if output_path.exists() and not force:
        print(f"[skip] {rel}  (exists, use --force to recompute)")
        return {"run_dir": str(run_dir), "skipped": True}

    result_files = sorted((run_dir / "step_results").glob("*.json"))
    if not result_files:
        print(f"[warn] no step result files in {rel}/step_results", file=sys.stderr)
        return {"run_dir": str(run_dir), "skipped": True}

    items_output: List[Dict[str, Any]] = []
    step_stats: Dict[int, Dict[str, Any]] = {}
    total_reward = 0.0
    total_steps = 0
    item_success_count = 0

    for result_path in tqdm(result_files, desc=str(rel)):
        result_data = json.loads(result_path.read_text(encoding="utf-8"))
        source_file: str = result_data.get("source_file", "")

        gt_result = _load_ground_truth(source_file, result_path)
        if gt_result is None:
            print(
                f"[warn] ground truth not found: {result_path.name}  "
                f"(source_file={source_file!r})",
                file=sys.stderr,
            )
            continue
        gt_path, gt_steps = gt_result

        steps_output: List[Dict[str, Any]] = []
        item_success = True

        for step in result_data.get("steps", []):
            step_num = step.get("step")
            gt_step = gt_steps.get(step_num, {})
            step_entry = compute_reward_for_step(
                step, gt_step, catalog,
                attribute_judge=attribute_judge,
                judge_context={
                    "task_id": result_data.get("task_id"),
                    "source_file": source_file,
                    "results_file": str(result_path),
                    "step": step_num,
                    "target_asin": gt_step.get("target_asin"),
                    "purchased_asin": step.get("purchased_asin"),
                },
            )
            steps_output.append(step_entry)
            total_reward += step_entry["reward"]
            total_steps += 1
            if step_entry["reward"] != 1.0:
                item_success = False
            _update_step_stats(step_stats, step_num, step_entry)

        if item_success:
            item_success_count += 1
        items_output.append({
            "task_id": result_data.get("task_id"),
            "source_file": source_file,
            "ground_truth_file": str(gt_path),
            "results_file": str(result_path),
            "steps": steps_output,
            "item_success": item_success,
        })

    total_items = len(items_output)
    average_reward = (total_reward / total_steps) if total_steps else 0.0
    item_success_rate = (item_success_count / total_items) if total_items else 0.0

    per_step_summary: Dict[str, Any] = {}
    for step_num, stats in sorted(step_stats.items()):
        count = stats["count"]
        per_step_summary[str(step_num)] = {
            "count": count,
            "average_reward": (stats["reward_sum"] / count) if count else 0.0,
            "success_rate": (stats["success_count"] / count) if count else 0.0,
            "failure_modes": {
                "no_purchase_ratio": (stats["no_purchase_count"] / count) if count else 0.0,
                "partial_reward_ratio": (stats["partial_reward_count"] / count) if count else 0.0,
                "other_failure_ratio": (stats["other_failure_count"] / count) if count else 0.0,
            },
            "rewards": stats.get("rewards", []),
        }

    per_step_success_rates = [e["success_rate"] for e in per_step_summary.values()]
    average_success_rate = (
        sum(per_step_success_rates) / len(per_step_success_rates)
        if per_step_success_rates else 0.0
    )

    output = {
        "summary_results": {
            "run_dir": str(run_dir),
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_items": total_items,
            "total_steps": total_steps,
            "average_reward": average_reward,
            "item_success_rate": item_success_rate,
            "item_success_count": item_success_count,
            "average_success_rate": average_success_rate,
            "per_step": per_step_summary,
        },
        "items": items_output,
    }

    output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[done] {rel}/{output_name}")
    return {
        "run_dir": str(run_dir),
        "skipped": False,
        "total_items": total_items,
        "total_steps": total_steps,
        "average_reward": average_reward,
        "item_success_rate": item_success_rate,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute LLM-judge reward reports for MemoryArena WebShop results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Target selection (mutually exclusive)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--run-dir", type=Path, metavar="DIR",
        help="Single run directory containing step_results/ "
             "(e.g. results/shopping/beauty/gpt-5-mini-split-long_context).",
    )
    target.add_argument(
        "--category", metavar="CAT",
        help="Process all run dirs under results/shopping/[CAT]/.",
    )
    target.add_argument(
        "--all", action="store_true",
        help="Process every run dir found under --results-dir.",
    )

    # Path overrides
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="Base results directory (default: MEMORYARENA_ROOT/results/shopping).",
    )
    parser.add_argument(
        "--product-catalog-dir", type=Path, default=None,
        help="Product catalog directory (default: data/shopping/product_catalog).",
    )
    parser.add_argument(
        "--domain-data", type=Path, default=None,
        help="domain_data.json path (default: data/shopping/domain_data.json).",
    )
    parser.add_argument(
        "--env-path", type=Path, default=None,
        help="Path to .env file with API key (auto-detected if omitted).",
    )

    # LLM judge options
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Disable LLM judge; use string-matching only.",
    )
    parser.add_argument(
        "--llm-model", default="gpt-4o",
        help="OpenAI model for attribute judging (default: gpt-4o).",
    )
    parser.add_argument(
        "--llm-max-retries", type=int, default=3,
        help="Max LLM attempts per attribute check (default: 3).",
    )
    parser.add_argument(
        "--llm-retry-delay", type=float, default=1.5,
        help="Seconds between LLM retries (default: 1.5).",
    )
    parser.add_argument(
        "--llm-output-name", default="llm_attribute_judgments.json",
        help="Filename for LLM judgment log saved in each run dir.",
    )

    # Output options
    parser.add_argument(
        "--output-name", default="reward_report.json",
        help="Reward report filename saved in each run dir (default: reward_report.json).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if reward_report.json already exists.",
    )

    args = parser.parse_args()

    def _abs(p: Optional[Path], default: Path) -> Path:
        if p is None:
            return default
        return p if p.is_absolute() else MEMORYARENA_ROOT / p

    results_base = _abs(args.results_dir, MEMORYARENA_ROOT / "results" / "shopping")
    product_catalog_dir = _abs(args.product_catalog_dir, default_product_catalog_dir())
    domain_data_path = _abs(args.domain_data, default_domain_data_path())
    env_path = args.env_path or _find_env_file()

    # Collect run directories
    if args.run_dir:
        run_dir = args.run_dir if args.run_dir.is_absolute() else MEMORYARENA_ROOT / args.run_dir
        if not (run_dir / "step_results").is_dir():
            raise SystemExit(f"step_results/ not found in: {run_dir}")
        run_dirs = [run_dir]
    elif args.category:
        run_dirs = collect_run_dirs(results_base, args.category)
        if not run_dirs:
            raise SystemExit(
                f"No run dirs found for category '{args.category}' under {results_base}"
            )
    else:
        run_dirs = collect_run_dirs(results_base)
        if not run_dirs:
            raise SystemExit(f"No run dirs found under {results_base}")

    print(f"Found {len(run_dirs)} run dir(s).")

    # Load product catalog (shared across all runs)
    if not product_catalog_dir.is_dir():
        raise SystemExit(f"Product catalog dir not found: {product_catalog_dir}")

    all_source_files: List[str] = []
    for rd in run_dirs:
        for rf in (rd / "step_results").glob("*.json"):
            try:
                sf = json.loads(rf.read_text(encoding="utf-8")).get("source_file", "")
                if sf:
                    all_source_files.append(sf)
            except Exception:
                pass
    catalog_files = select_catalog_files(product_catalog_dir, domain_data_path, all_source_files)
    catalog_desc = (
        f"{len(catalog_files)} file(s)" if catalog_files is not None else "all files"
    )
    print(f"Loading catalog ({catalog_desc})...")
    catalog = load_catalog(product_catalog_dir, catalog_files)

    # Create LLM judge (optional)
    attribute_judge: Optional[AttributeJudge] = None
    if not args.no_llm:
        try:
            attribute_judge = create_attribute_judge(
                env_path=env_path,
                model=args.llm_model,
                max_retries=args.llm_max_retries,
                retry_delay=args.llm_retry_delay,
            )
            print(f"LLM judge: {attribute_judge.backend} / {attribute_judge.model}")
        except Exception as exc:
            print(
                f"[warn] LLM judge unavailable ({exc}); falling back to string matching.",
                file=sys.stderr,
            )

    # Process each run directory
    all_summaries: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        summary = process_run_dir(
            run_dir=run_dir,
            catalog=catalog,
            attribute_judge=attribute_judge,
            output_name=args.output_name,
            force=args.force,
        )
        all_summaries.append(summary)

        # Save per-run LLM judgment log and reset records for next run
        if attribute_judge is not None and not summary.get("skipped"):
            llm_out = run_dir / args.llm_output_name
            llm_out.write_text(
                json.dumps(
                    attribute_judge.build_output({
                        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "run_dir": str(run_dir),
                    }),
                    ensure_ascii=True, indent=2,
                ),
                encoding="utf-8",
            )
            attribute_judge.records = []  # reset for next run; cache is kept

    # Print aggregate summary
    processed = [s for s in all_summaries if not s.get("skipped")]
    if processed:
        print(f"\n{'='*60}")
        print(f"Processed {len(processed)} run(s):\n")
        for s in processed:
            rel = display_path(Path(s["run_dir"]))
            print(
                f"  {rel}\n"
                f"    steps={s['total_steps']}  "
                f"avg_reward={s['average_reward']:.3f}  "
                f"item_success={s['item_success_rate']:.1%}\n"
            )


if __name__ == "__main__":
    main()
