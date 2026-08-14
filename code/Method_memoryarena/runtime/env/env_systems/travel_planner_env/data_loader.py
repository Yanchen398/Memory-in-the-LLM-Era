import re
from datasets import load_dataset

DATASET_NAME = "ZexueHe/memoryarena"
DATASET_CONFIG = "group_travel_planner"


def _extract_name(query_str):
    m = re.match(r"I am (\w+)\.", query_str)
    return m.group(1) if m else "Person"


def _convert_row(row):
    """Convert a HuggingFace dataset row to the format expected by the codebase."""
    questions = []
    for i, query_str in enumerate(row["questions"], start=1):
        questions.append({
            "round_idx": i,
            "name": _extract_name(query_str),
            "query": query_str,
        })

    answers = []
    for i, daily_plans in enumerate(row["answers"], start=1):
        answers.append({
            "round_idx": i,
            "daily_plans": [dict(d) for d in daily_plans],
        })

    base_person = row["base_person"]
    return {
        "id": row["id"],
        "base_person": {
            "name": base_person["name"],
            "query": base_person["query"],
            "daily_plans": [dict(d) for d in base_person["daily_plans"]],
        },
        "questions": questions,
        "answers": answers,
    }


def load_travel_data():
    """Load travel planner data from HuggingFace and convert to internal format."""
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
    return [_convert_row(row) for row in ds]
