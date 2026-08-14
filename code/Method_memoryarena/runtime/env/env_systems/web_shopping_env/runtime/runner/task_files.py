from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ..runtime_paths import MEMORYARENA_ROOT

HF_DATASET_ID = "ZexueHe/memoryarena"
HF_CONFIG_NAME = "bundled_shopping"
HF_SPLIT = "test"



def _reconstruct_task_def_from_hf_row(row: Dict) -> Dict:
    """Reconstruct a complete task_def dict from an HF *bundled_shopping* row.

    The HF schema only provides four fields per row:

    * ``id``        – global integer index
    * ``questions`` – list of per-step agent instructions (one per product)
    * ``answers``   – list of ``{target_asin, attributes}`` dicts
    * ``category``  – string like ``"baking_item_0"``

    Each ``questions[i]`` already contains the global rules preamble followed by
    ``"Product {i+1}: ..."`` body.  We rebuild the single multi-product
    ``agent_instruction`` by extracting the shared prefix from ``questions[0]``
    and concatenating the individual product sections; ``split_agent_instruction``
    can then split it back correctly.
    """
    questions: List[str] = row["questions"]
    answers: List[Dict] = row["answers"]
    category: str = row["category"]

    # --- Reconstruct agent_instruction -----------------------------------
    # Extract the global-rules prefix that appears before "Product 1:" in
    # the first question.
    prefix_match = re.search(r"Product\s+1:", questions[0], re.IGNORECASE)
    prefix = questions[0][: prefix_match.start()].rstrip() if prefix_match else ""

    # Extract "Product N: <body>" from each question.
    product_sections: List[str] = []
    for i, q in enumerate(questions):
        m = re.search(rf"Product\s+{i + 1}:", q, re.IGNORECASE)
        product_sections.append(q[m.start() :].strip() if m else f"Product {i + 1}:\n")

    # The extracted ``prefix`` already ends with the 64-dash separator line
    # (rstripped, so no trailing newline).  We only need "\n" to connect it
    # to the first product section; subsequent sections are joined with the
    # full separator so that split_agent_instruction() produces section bodies
    # byte-for-byte identical to those in locally-stored item_*.json files.
    separator = "\n\n" + "-" * 64 + "\n"
    agent_instruction = (prefix + "\n" + separator.join(product_sections)).strip() + "\n"

    # --- Reconstruct steps -----------------------------------------------
    num_steps = len(questions)
    steps = [
        {
            "step": i + 1,
            "target_asin": answers[i]["target_asin"],
            "step_description": f"Product {i + 1}",
            "requirements": {
                "product_type": "",
                "attributes": answers[i].get("attributes", []),
                "options": {},
                "price_constraints": {},
            },
        }
        for i in range(num_steps)
    ]

    return {
        "task_id": category,
        "task_type": "bundled_shopping",
        "task_description": f"Bundle shopping task: {category}",
        "source_case": "",
        "agent_instruction": agent_instruction,
        "steps": steps,
        "global_constraints": {
            "total_expense_upper": None,
            "max_steps": num_steps,
            "purchase_order": "sequence",
        },
        "target_products": [ans["target_asin"] for ans in answers],
        "evaluation_criteria": {
            **{f"step_{i + 1}_success": {} for i in range(num_steps)},
            "overall_success": {"all_steps_completed": True},
        },
        "metadata": {
            "hf_id": row.get("id"),
            "hf_category": category,
        },
    }


def collect_task_files_from_hf(
    category_prefix: str,
    limit: int | None = None,
) -> list[Path]:
    """Download tasks from the HuggingFace *bundled_shopping* config, filter by
    category prefix, reconstruct full task_def dicts, persist them to a local
    cache directory, and return the resulting file paths.

    Parameters
    ----------
    category_prefix:
        The leading token of the HF ``category`` field, e.g. ``"baking"``.
        Rows whose ``category`` starts with ``"{prefix}_item_"`` are selected.
    limit:
        Maximum number of task files to return.  ``None`` means no limit.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace task loading. "
            "Install it with:  pip install datasets"
        ) from exc

    ds = load_dataset(HF_DATASET_ID, HF_CONFIG_NAME, split=HF_SPLIT)

    prefix_pattern = f"{category_prefix}_item_"
    rows = [row for row in ds if row["category"].startswith(prefix_pattern)]

    def _item_number(row: Dict) -> int:
        m = re.search(r"_item_(\d+)$", row["category"])
        return int(m.group(1)) if m else 0

    rows = sorted(rows, key=_item_number)
    if limit is not None:
        rows = rows[:limit]

    if not rows:
        raise FileNotFoundError(
            f"No HF tasks found for category prefix '{category_prefix}' "
            f"in dataset {HF_DATASET_ID!r} / config {HF_CONFIG_NAME!r}."
        )

    cache_dir = MEMORYARENA_ROOT / "hf_task_cache" / category_prefix
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for row in rows:
        task_def = _reconstruct_task_def_from_hf_row(row)
        item_num = _item_number(row)
        out_path = cache_dir / f"item_{item_num}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(task_def, f, ensure_ascii=False, indent=2)
        paths.append(out_path)

    return paths


def collect_all_hf_category_prefixes() -> list[str]:
    """Return all unique category prefixes present in the HF *bundled_shopping* split.

    E.g. ``["baking", "beauty", "electronics", "grocery", "home_decor", "mens_outfit"]``.
    The order reflects the first occurrence of each prefix in the dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace task loading. "
            "Install it with:  pip install datasets"
        ) from exc

    ds = load_dataset(HF_DATASET_ID, HF_CONFIG_NAME, split=HF_SPLIT)
    seen: set[str] = set()
    prefixes: list[str] = []
    for row in ds:
        m = re.match(r"^([^_]+)_item_\d+$", row["category"])
        if m:
            p = m.group(1)
            if p not in seen:
                seen.add(p)
                prefixes.append(p)
    return prefixes



def split_agent_instruction(agent_instruction: str) -> tuple[str, List[Dict[str, str]]]:
    """
    Split agent_instruction into prefix and per-product sections based on 'Product X:' markers.
    """
    pattern = re.compile(r"(Product\s+(\d+):)", re.IGNORECASE)
    matches = list(pattern.finditer(agent_instruction))
    if not matches:
        return agent_instruction.strip(), []

    prefix = agent_instruction[: matches[0].start()].strip()
    sections: List[Dict[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(agent_instruction)
        body = agent_instruction[start:end].strip()
        sections.append(
            {"step": int(match.group(2)), "header": match.group(1), "body": body}
        )
    return prefix, sections


def build_instruction_for_step(
    prefix: str,
    sections: List[Dict[str, str]],
    current_step: int,
    feedback: Dict[int, str],
    include_history: bool = True,
) -> str:
    """
    Construct the agent_instruction for a specific step.

    - Keeps prefix content
    - For completed steps, replaces the original body with the feedback string
    - For the current step, keeps the original body
    - Ignores future steps
    """
    parts: List[str] = []
    if prefix:
        parts.append(prefix)

    history_parts: List[str] = []
    current_parts: List[str] = []

    for section in sections:
        if section["step"] > current_step:
            break
        if section["step"] < current_step:
            body = feedback.get(section["step"], section["body"])
            history_parts.append(f"{section['header']}\n{body}".strip())
        else:
            current_parts.append(f"{section['header']}\n{section['body']}".strip())

    if include_history and history_parts:
        parts.append("*** Purchase History ***\n" + "\n\n".join(history_parts))
    if current_parts:
        reminder = (
            "You only need to buy the item for this step; use the ground-truth items "
            "in Purchase History as references and follow the requirements to buy the correct item."
        )
        parts.append(
            "*** What You Need to Buy at Current Step ***\n"
            + reminder
            + "\n\n"
            + "\n\n".join(current_parts)
        )

    return "\n\n".join(part.strip() for part in parts if part).strip() + "\n"


def build_single_step_task(
    task_def: Dict,
    step_idx: int,
    agent_instruction: str,
    total_steps: int | None = None,
) -> Dict:
    """
    Create a single-step task definition for the specified step index (0-based).
    """
    single_step = copy.deepcopy(task_def)
    step = single_step["steps"][step_idx]
    single_step["task_id"] = f"{task_def['task_id']}_step_{step_idx + 1}"
    step_total = total_steps if total_steps is not None else len(task_def["steps"])
    single_step["task_description"] = (
        f"{task_def.get('task_description', '')} (step {step_idx + 1}/{step_total})"
    )
    single_step["agent_instruction"] = agent_instruction
    single_step["steps"] = [step]
    single_step["target_products"] = [step["target_asin"]]

    global_constraints = copy.deepcopy(task_def.get("global_constraints", {}))
    global_constraints["max_steps"] = 1
    global_constraints["purchase_order"] = "single"
    single_step["global_constraints"] = global_constraints

    eval_criteria = task_def.get("evaluation_criteria", {})
    key = f"step_{step_idx + 1}_success"
    single_step["evaluation_criteria"] = {
        key: eval_criteria.get(key, {}),
        "overall_success": {"all_steps_completed": True},
    }

    return single_step


def write_temp_task_file(task_def: Dict, base_dir: Path) -> Path:
    """Persist a task definition to a temporary file for the environment client."""
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = base_dir / f"{task_def['task_id']}_{timestamp}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(task_def, f, ensure_ascii=False, indent=2)
    return file_path
