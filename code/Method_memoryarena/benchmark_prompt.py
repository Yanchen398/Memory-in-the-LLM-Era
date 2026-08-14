"""Shared benchmark prompts for LOCOMO and LongMemEval.

Memory methods keep their native extraction, management, and retrieval logic.
The paper keeps method-specific response prompts, then shares answer
simplification and LLM-judge prompts across methods.  The optional answer
builder below is for explicitly named prompt-ablation runs only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


SUPPORTED_DATASETS = frozenset({"locomo", "longmemeval"})
DEFAULT_CONTEXT_CHAR_LIMIT = 16_000
UNKNOWN_ANSWER = "I don't know."


ANSWER_SIMPLIFICATION_TEMPLATE = """Your task is to act as an answer simplifier. I will give you a question and a full-sentence answer. You must reduce the answer to its most critical component.
Follow these rules:
1. **Extract the Core Information:** Identify the primary piece of information that directly answers the question.
2. **Remove Extraneous Phrases:** Eliminate phrases like "Based on the information provided...", "The answer is...", "As per the document...", etc.
3. **Omit Explanations:** Do not include any justifications, reasoning, or additional context from the original answer.
4. **Be Concise:** The output should be the shortest possible string that still accurately answers the question.

Example:
- Question: "What degree did I graduate with?"
- Original Answer: "Based on the information provided, you graduated with a degree in Business Administration."
- Simplified Answer: "Business Administration"

Here is the question and answer to simplify:
- Question: {question}
- Original Answer: {answer}
- Simplified Answer:
"""


LOCOMO_JUDGE_TEMPLATE = """Your task is to label an answer to a question as CORRECT or WRONG. You will be given the following data: (1) a question posed by one user to another user, (2) a gold ground-truth answer, and (3) a generated answer which you will score as CORRECT or WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be concise and short. The generated answer might be much longer, but grade generously: as long as it touches on the same topic as the gold answer, count it as CORRECT.

For time-related questions, the gold answer may be a specific date, month, or year. The generated answer may be longer or use relative time references. Count it as CORRECT when it refers to the same date or time period, even if the format differs.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Return JSON only with exactly one key named "label" whose value is either "CORRECT" or "WRONG". Do not include both labels.
"""


def build_simplification_prompt(*, question: Any, answer: Any) -> str:
    clean_question = str(question or "").strip()
    clean_answer = str(answer or "").strip()
    if not clean_question:
        raise ValueError("question must be non-empty")
    if not clean_answer:
        raise ValueError("answer must be non-empty")
    return ANSWER_SIMPLIFICATION_TEMPLATE.format(
        question=clean_question,
        answer=clean_answer,
    )


def build_locomo_judge_prompt(
    *, question: Any, gold_answer: Any, generated_answer: Any
) -> str:
    values = {
        "question": str(question or "").strip(),
        "gold_answer": str(gold_answer or "").strip(),
        "generated_answer": str(generated_answer or "").strip(),
    }
    empty = [name for name, value in values.items() if not value]
    if empty:
        raise ValueError(f"judge fields must be non-empty: {', '.join(empty)}")
    return LOCOMO_JUDGE_TEMPLATE.format(**values)


def _coerce_memory_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        lines = []
        for key, item in value.items():
            text = _coerce_memory_text(item)
            if text:
                lines.append(f"[{key}]\n{text}")
        return "\n".join(lines)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        lines = []
        for rank, item in enumerate(value, start=1):
            text = _coerce_memory_text(item)
            if text:
                lines.append(f"[{rank}] {text}")
        return "\n".join(lines)
    return str(value).strip()


def render_memory_context(
    memories: Any,
    *,
    char_limit: int = DEFAULT_CONTEXT_CHAR_LIMIT,
) -> str:
    """Render retrieved memories deterministically without changing their order."""

    if int(char_limit) <= 0:
        raise ValueError("char_limit must be positive")
    text = _coerce_memory_text(memories)
    if not text:
        return "(no retrieved memory)"
    if len(text) <= int(char_limit):
        return text
    return text[: int(char_limit)].rstrip() + "\n[context truncated]"


def build_answer_messages(
    *,
    dataset: str,
    question: str,
    memories: Any,
    question_date: Any = None,
    context_char_limit: int = DEFAULT_CONTEXT_CHAR_LIMIT,
) -> list[dict[str, str]]:
    """Build an optional shared answer prompt for explicit ablation runs.

    Paper-faithful baseline runs must retain their native response prompt and
    apply :func:`build_simplification_prompt` afterward.
    """

    normalized_dataset = str(dataset or "").strip().lower().replace("-", "")
    if normalized_dataset == "longmemeval":
        dataset_name = "longmemeval"
    elif normalized_dataset == "locomo":
        dataset_name = "locomo"
    else:
        raise ValueError(
            f"dataset must be one of {sorted(SUPPORTED_DATASETS)}, got {dataset!r}"
        )

    clean_question = str(question or "").strip()
    if not clean_question:
        raise ValueError("question must be non-empty")

    context = render_memory_context(memories, char_limit=context_char_limit)
    date_text = str(question_date or "").strip() or "not provided"
    dataset_rule = (
        "Return only a short direct answer, normally no more than six words."
        if dataset_name == "locomo"
        else "Return only the concise direct answer required by the question."
    )
    system = (
        "You answer questions using only the retrieved memory evidence. "
        "Do not use outside knowledge, do not invent missing facts, and do not "
        "describe your reasoning or the retrieval process. Resolve relative dates "
        "against the question date and prefer event timestamps over narration dates. "
        f"If the evidence is insufficient, answer exactly: {UNKNOWN_ANSWER} "
        f"{dataset_rule}"
    )
    user = (
        f"Dataset: {dataset_name}\n"
        f"Question date: {date_text}\n"
        f"Retrieved memory evidence:\n{context}\n\n"
        f"Question: {clean_question}\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_answer_prompt(**kwargs: Any) -> str:
    """Flatten :func:`build_answer_messages` for single-string model APIs."""

    messages = build_answer_messages(**kwargs)
    return f"{messages[0]['content']}\n\n{messages[1]['content']}"
