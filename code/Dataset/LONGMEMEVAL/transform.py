import argparse
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_INPUT_FILENAME = "longmemeval_s.json"
DEFAULT_OUTPUT_SUFFIX = "_locomo"
ROLE_MAP = {
    "user": "User",
    "assistant": "Assistant",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LongMemEval JSON files into the LOCOMO-aligned schema used by this repository."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_FILENAME,
        help="Path to the source LongMemEval JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the converted output JSON file. Defaults to '<input>_locomo.json'.",
    )
    return parser.parse_args()


def resolve_default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}{DEFAULT_OUTPUT_SUFFIX}.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def normalize_speaker(role: str) -> str:
    normalized = str(role or "").strip().lower()
    if normalized in ROLE_MAP:
        return ROLE_MAP[normalized]
    if not normalized:
        return "Unknown"
    return normalized.replace("_", " ").title()


def validate_record(record: Dict):
    required_keys = [
        "question_id",
        "question_type",
        "question",
        "answer",
        "haystack_dates",
        "haystack_session_ids",
        "haystack_sessions",
    ]
    missing = [key for key in required_keys if key not in record]
    if missing:
        raise ValueError(f"Missing required keys in LongMemEval record {record.get('question_id')}: {missing}")

    session_count = len(record["haystack_sessions"])
    if len(record["haystack_dates"]) != session_count or len(record["haystack_session_ids"]) != session_count:
        raise ValueError(
            f"LongMemEval record {record.get('question_id')} has mismatched haystack array lengths: "
            f"dates={len(record['haystack_dates'])}, "
            f"session_ids={len(record['haystack_session_ids'])}, "
            f"sessions={session_count}"
        )


def convert_session(session_index: int, turns: List[Dict]) -> List[Dict]:
    converted_turns = []
    for turn_index, turn in enumerate(turns, start=1):
        converted_turns.append(
            {
                "speaker": normalize_speaker(turn.get("role", "")),
                "dia_id": f"D{session_index}:{turn_index}",
                "text": str(turn.get("content", "")),
            }
        )
    return converted_turns


def convert_record(record: Dict) -> Dict:
    validate_record(record)

    conversation = {
        "speaker_a": "User",
        "speaker_b": "Assistant",
    }

    for session_index, (session_date, turns) in enumerate(
        zip(record["haystack_dates"], record["haystack_sessions"]),
        start=1,
    ):
        conversation[f"session_{session_index}_date_time"] = str(session_date)
        conversation[f"session_{session_index}"] = convert_session(session_index, turns)

    qa_entry = {
        "question": str(record.get("question", "")),
        "answer": str(record.get("answer", "")),
        "evidence": [str(item) for item in record.get("answer_session_ids", [])],
        "category": record.get("question_type"),
    }

    return {
        "sample_id": str(record.get("question_id")),
        "qa": [qa_entry],
        "conversation": conversation,
    }


def convert_dataset(records: List[Dict]) -> List[Dict]:
    if not isinstance(records, list):
        raise ValueError("Expected the LongMemEval source file to contain a JSON array.")
    return [convert_record(record) for record in records]


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path

    output_path = Path(args.output) if args.output else resolve_default_output_path(input_path)
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    records = load_json(input_path)
    converted = convert_dataset(records)
    write_json(output_path, converted)

    print(f"Converted {len(converted)} LongMemEval records.")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
