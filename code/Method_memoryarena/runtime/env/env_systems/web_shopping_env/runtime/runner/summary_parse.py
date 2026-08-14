from __future__ import annotations

import re
from typing import Dict


def normalize_message_fields(message: dict) -> dict:
    """Normalize conversation message keys across different schemas."""
    return {
        "role": message.get("role") or message.get("from") or "",
        "content": message.get("content") or message.get("value") or "",
    }


def extract_final_env_state(conversation: list[dict]) -> str:
    """Get the last environment message (state) from the conversation."""
    for message in reversed(conversation):
        normalized = normalize_message_fields(message)
        if normalized["role"] in ("user", "human"):
            return normalized["content"]
    return ""


def parse_combo_summary(final_state: str) -> dict:
    """
    Parse the combo summary text to extract purchased/target pairs or step rewards.

    Returns a dict with either:
      step_rewards: list of {idx, asin, price, reward, total_reward?}
      or the legacy fields:
        targets: list of {idx, desc, purchased_asin, expected_asin, match_score, reward, purchased_price}
        unmatched: list of {idx, asin, price}
    """
    step_rewards = []
    total_reward = None

    for raw_line in final_state.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m_step = re.search(
            r"#\s*(\d+)\s+ASIN=([A-Z0-9]+|UNKNOWN)\s+Price=\$?([\d\.,]+|N/A)\s+Reward=([+\-]?\d+\.?\d*)",
            line,
            re.IGNORECASE,
        )
        if m_step:
            price_raw = m_step.group(3)
            price_val = None
            if price_raw and price_raw.upper() != "N/A":
                try:
                    price_val = float(price_raw.replace(",", "").lstrip("$"))
                except ValueError:
                    price_val = None

            try:
                reward_val = float(m_step.group(4))
            except ValueError:
                reward_val = None

            step_rewards.append(
                {
                    "idx": int(m_step.group(1)),
                    "asin": m_step.group(2),
                    "price": price_val,
                    "reward": reward_val,
                }
            )
            continue

        if total_reward is None:
            m_total = re.search(r"Total Reward:\s*([+\-]?\d+\.?\d*)", line, re.IGNORECASE)
            if m_total:
                try:
                    total_reward = float(m_total.group(1))
                except ValueError:
                    total_reward = None

    if step_rewards:
        return {"step_rewards": step_rewards, "total_reward": total_reward}

    targets = []
    unmatched = []
    current = None

    for raw_line in final_state.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m_target = re.match(r"Target\s+(\d+):\s*(.*)", line, re.IGNORECASE)
        if m_target:
            if current:
                targets.append(current)
            current = {
                "idx": int(m_target.group(1)),
                "desc": m_target.group(2).strip(),
                "purchased_asin": None,
                "purchased_price": None,
                "expected_asin": None,
                "match_score": None,
                "reward": None,
            }
            continue

        if current:
            m_purchase = re.search(r"Purchased:\s*([A-Z0-9]{10})(?:\s*\(\$([\d\.,]+)\))?", line)
            if m_purchase:
                current["purchased_asin"] = m_purchase.group(1)
                if m_purchase.lastindex and m_purchase.group(2):
                    try:
                        current["purchased_price"] = float(
                            m_purchase.group(2).replace(",", "")
                        )
                    except ValueError:
                        current["purchased_price"] = None

            m_expected = re.search(r"Expected:\s*([A-Z0-9]{10})", line)
            if m_expected:
                current["expected_asin"] = m_expected.group(1)

            m_score = re.search(r"Match Score:\s*([\d\.]+)%", line)
            if m_score:
                try:
                    current["match_score"] = float(m_score.group(1))
                except ValueError:
                    pass

            m_reward = re.search(r"Reward:\s*([+\-]?\d+\.?\d*)", line)
            if m_reward:
                try:
                    current["reward"] = float(m_reward.group(1))
                except ValueError:
                    pass

        m_unmatched = re.search(r"Purchase #(\d+):\s*([A-Z0-9]{10})\s*\(\$([\d\.,]+)\)", line)
        if m_unmatched:
            try:
                price_val = float(m_unmatched.group(3).replace(",", ""))
            except ValueError:
                price_val = None
            unmatched.append(
                {
                    "idx": int(m_unmatched.group(1)),
                    "asin": m_unmatched.group(2),
                    "price": price_val,
                }
            )

    if current:
        targets.append(current)

    return {"targets": targets, "unmatched": unmatched}


def extract_purchase_from_summary(parsed_summary: dict) -> tuple[str | None, float | None]:
    """
    Extract purchased ASIN and price from parsed summary dict produced by parse_combo_summary.
    """
    step_rewards = parsed_summary.get("step_rewards")
    if step_rewards:
        first = step_rewards[0]
        return first.get("asin"), first.get("price")

    for target in parsed_summary.get("targets", []):
        if target.get("purchased_asin"):
            return target.get("purchased_asin"), target.get("purchased_price")
    if parsed_summary.get("unmatched"):
        first = parsed_summary["unmatched"][0]
        return first.get("asin"), first.get("price")
    return None, None


def index_conversation_by_asin(conversation: list[dict]) -> dict[str, str]:
    """Build a lookup from ASIN to the first environment message that contains it."""
    asin_to_text: dict[str, str] = {}
    for msg in conversation:
        normalized = normalize_message_fields(msg)
        content = normalized["content"] or ""
        for asin in re.findall(r"\b[A-Z0-9]{10}\b", content):
            if asin not in asin_to_text:
                asin_to_text[asin] = content
    return asin_to_text


def extract_product_name_from_text(asin: str, text: str) -> str | None:
    """Heuristically extract a product name that appears near the ASIN inside the given text."""
    if not text:
        return None

    segments = [seg.strip() for seg in re.split(r"\[SEP\]|\\n|\n", text) if seg.strip()]
    for idx, seg in enumerate(segments):
        if asin in seg:
            if idx + 1 < len(segments):
                candidate = segments[idx + 1]
                if not re.fullmatch(r"\$?[\d\.,]+", candidate):
                    return candidate

    for seg in segments:
        if asin in seg:
            cleaned = seg.replace(asin, "").strip(" -:")
            if cleaned:
                return cleaned
    return None
