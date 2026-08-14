from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

from ..runtime_paths import default_items_file
from .summary_parse import (
    extract_final_env_state,
    extract_product_name_from_text,
    extract_purchase_from_summary,
    index_conversation_by_asin,
    parse_combo_summary,
)


@lru_cache(maxsize=1)
def load_product_catalog(product_data_path: str | None = None) -> dict[str, dict]:
    """Load product catalog keyed by ASIN. Cached to avoid repeated disk reads."""
    try:
        if product_data_path is None:
            product_data_path = os.getenv(
                "COMBO_WEBSHOP_PRODUCT_DATA",
                str(default_items_file()),
            )
        path = Path(product_data_path)
        if not path.exists():
            print(f"[Feedback] Product data not found at {path}, skipping name lookup")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            products = json.load(f)
        catalog = {}
        for product in products:
            asin = product.get("asin")
            if not asin:
                continue
            catalog[asin] = product
        print(f"[Feedback] Loaded product catalog with {len(catalog)} entries from {path}")
        return catalog
    except Exception as exc:
        print(f"[Feedback] Failed to load product catalog: {exc}")
        return {}


def get_product_name_from_catalog(asin: str) -> str | None:
    """Lookup product name from cached catalog, if available."""
    if not asin:
        return None
    catalog = load_product_catalog()
    entry = catalog.get(asin)
    if not entry:
        return None
    return entry.get("name") or entry.get("title") or entry.get("small_description")


def format_feedback(
    step_num: int,
    purchase_asin: str | None,
    price: float | None,
    target_asin: str,
    purchase_name: str | None,
    target_name: str | None,
) -> str:
    """Create a short feedback string to replace prior product instructions."""
    price_text = f"${price:.2f}" if isinstance(price, (int, float)) else "price unknown"
    asin_text = purchase_asin or "no purchase"
    match_text = "matched target" if purchase_asin == target_asin else "did NOT match target"
    name_text = purchase_name or "name unknown"
    target_name_text = target_name or "name unknown"
    return (
        f"Product {step_num} shopping result: bought {asin_text} ({name_text}) at {price_text}; "
        f"{match_text} (ground truth: {target_asin} ({target_name_text}))."
    )


def summarize_step_result(exps, step_def: Dict, step_num: int) -> Dict:
    """Extract purchased ASIN/price and match status from experiences."""
    summary: Dict[str, object] = {
        "step": step_num,
        "target_asin": step_def["target_asin"],
        "target_description": step_def.get("step_description", ""),
        "target_name": get_product_name_from_catalog(step_def["target_asin"])
        or step_def.get("step_description", ""),
        "purchased_asin": None,
        "purchased_name": None,
        "purchased_price": None,
        "match_ground_truth": False,
        "final_state": "",
        "reward": None,
    }

    if not exps.experiences:
        return summary

    exp = exps.experiences[0]
    summary["reward"] = getattr(exp, "reward", None)
    final_state = extract_final_env_state(exp.conversation)
    summary["final_state"] = final_state
    parsed = parse_combo_summary(final_state)
    purchased_asin, purchased_price = extract_purchase_from_summary(parsed)
    summary["purchased_asin"] = purchased_asin
    summary["purchased_price"] = purchased_price
    if purchased_asin:
        asin_text_lookup = index_conversation_by_asin(exp.conversation)
        summary["purchased_name"] = extract_product_name_from_text(
            purchased_asin, asin_text_lookup.get(purchased_asin, "")
        ) or get_product_name_from_catalog(purchased_asin)
    if purchased_asin:
        summary["match_ground_truth"] = purchased_asin == step_def["target_asin"]
    return summary


def summarize_all_steps_from_final_state(
    task_def: Dict,
    final_state: str,
    conversation: list[dict] | None = None,
    total_steps: int | None = None,
) -> list[Dict]:
    """Summarize all steps from a single final environment state."""
    step_defs = task_def.get("steps", [])
    if total_steps is not None:
        step_defs = step_defs[:total_steps]

    parsed = parse_combo_summary(final_state) if final_state else {}
    step_rewards = parsed.get("step_rewards") or []
    targets = parsed.get("targets") or []

    reward_lookup = {entry.get("idx"): entry for entry in step_rewards if entry.get("idx")}
    target_lookup = {entry.get("idx"): entry for entry in targets if entry.get("idx")}

    asin_text_lookup = index_conversation_by_asin(conversation) if conversation else {}

    summaries: list[Dict] = []
    for idx, step_def in enumerate(step_defs):
        step_num = idx + 1
        summary: Dict[str, object] = {
            "step": step_num,
            "target_asin": step_def["target_asin"],
            "target_description": step_def.get("step_description", ""),
            "target_name": get_product_name_from_catalog(step_def["target_asin"])
            or step_def.get("step_description", ""),
            "purchased_asin": None,
            "purchased_name": None,
            "purchased_price": None,
            "match_ground_truth": False,
            "final_state": final_state,
            "reward": None,
        }

        reward_entry = reward_lookup.get(step_num)
        target_entry = target_lookup.get(step_num)

        purchased_asin = None
        purchased_price = None
        reward = None

        if reward_entry:
            purchased_asin = reward_entry.get("asin")
            purchased_price = reward_entry.get("price")
            reward = reward_entry.get("reward")
        elif target_entry:
            purchased_asin = target_entry.get("purchased_asin")
            purchased_price = target_entry.get("purchased_price")
            reward = target_entry.get("reward")

        if purchased_asin and str(purchased_asin).upper() == "UNKNOWN":
            purchased_asin = None

        summary["purchased_asin"] = purchased_asin
        summary["purchased_price"] = purchased_price
        summary["reward"] = reward

        if purchased_asin:
            summary["purchased_name"] = extract_product_name_from_text(
                purchased_asin, asin_text_lookup.get(purchased_asin, "")
            ) or get_product_name_from_catalog(purchased_asin)
            summary["match_ground_truth"] = purchased_asin == step_def["target_asin"]

        summary["feedback"] = format_feedback(
            step_num,
            summary.get("purchased_asin"),
            summary.get("purchased_price"),
            summary.get("target_asin"),
            summary.get("purchased_name"),
            summary.get("target_name"),
        )

        summaries.append(summary)

    return summaries


def summarize_step_from_final_state(step_def: Dict, step_num: int, final_state: str) -> Dict:
    """Rebuild a step summary from a saved log's final environment state."""
    summary: Dict[str, object] = {
        "step": step_num,
        "target_asin": step_def["target_asin"],
        "target_description": step_def.get("step_description", ""),
        "target_name": get_product_name_from_catalog(step_def["target_asin"])
        or step_def.get("step_description", ""),
        "purchased_asin": None,
        "purchased_name": None,
        "purchased_price": None,
        "match_ground_truth": False,
        "final_state": final_state,
        "reward": None,
    }

    if not final_state:
        return summary

    parsed = parse_combo_summary(final_state)
    purchased_asin, purchased_price = extract_purchase_from_summary(parsed)
    summary["purchased_asin"] = purchased_asin
    summary["purchased_price"] = purchased_price
    if purchased_asin:
        summary["purchased_name"] = get_product_name_from_catalog(purchased_asin)
        summary["match_ground_truth"] = purchased_asin == step_def["target_asin"]
    return summary


def hydrate_step_summary(step_summary: Dict, step_def: Dict, step_num: int) -> Dict:
    """Ensure a step summary has all fields and feedback populated."""
    summary = dict(step_summary)
    summary.setdefault("step", step_num)
    summary.setdefault("target_asin", step_def["target_asin"])
    summary.setdefault("target_description", step_def.get("step_description", ""))

    target_name = summary.get("target_name") or get_product_name_from_catalog(step_def["target_asin"])
    if not target_name:
        target_name = step_def.get("step_description", "")
    summary["target_name"] = target_name

    purchased_asin = summary.get("purchased_asin")
    purchased_price = summary.get("purchased_price")
    purchased_name = summary.get("purchased_name")

    if purchased_asin and not purchased_name:
        summary["purchased_name"] = get_product_name_from_catalog(purchased_asin)

    if purchased_asin is not None and summary.get("match_ground_truth") is None:
        summary["match_ground_truth"] = purchased_asin == step_def["target_asin"]

    if not summary.get("feedback"):
        summary["feedback"] = format_feedback(
            step_num,
            purchased_asin,
            purchased_price,
            summary["target_asin"],
            summary.get("purchased_name"),
            summary.get("target_name"),
        )

    return summary
