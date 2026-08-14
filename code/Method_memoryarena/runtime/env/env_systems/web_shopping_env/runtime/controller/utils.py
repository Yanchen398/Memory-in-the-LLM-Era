from __future__ import annotations

import json
import re
from typing import Sequence

from .types import ActionFormat, ActionWithTought, ConversationMessage


INVOKING_FUNCTION_PROMPT = """

If you want to invoke a provided function or tool, please reply in the following *JSON* format:
```json
{
    "thought": "I think ...",
    "function_name": "function_name",
    "arguments": <valid json object of args>
}
```
Only reply the *JSON* object, no other text should be present.
"""

WRITE_CODE_PROMPT = """

If you want to call these functions, please reply the python code block:
```python
# Write you thought in the code comment before you call any function.
<write valid python code here.>
```
Only reply the code block with "```python" and "```",  no other text should be present.
"""


def format_function_call_prompt(function_description: Sequence[dict]) -> str:
    prompt = "You have the following functions available:\n\n"
    tool_descs = [{"type": "function", "function": f} for f in function_description]
    prompt += "\n".join(
        json.dumps(f, ensure_ascii=False, indent=2) for f in tool_descs
    )
    prompt += INVOKING_FUNCTION_PROMPT
    return prompt


def generate_function_signatures(function_descriptions: Sequence[dict]) -> str:
    function_strings = []
    for func in function_descriptions:
        name = func["name"]
        description = func["description"]
        params = func["parameters"]["properties"]
        required_params = func["parameters"].get("required", [])
        signature_params = ", ".join(
            [
                f"{param}='{param}'" if param not in required_params else param
                for param in params
            ]
        )
        function_signature = f"def {name}({signature_params}):"
        docstring = f'    """\n    {description}\n\n'
        for param, details in params.items():
            docstring += (
                f"    :param {param} ({details['type']}): {details['description']}\n"
            )
        docstring += '    """'
        function_strings.append(f"{function_signature}\n{docstring}\n")
    return "\n".join(function_strings)


def format_code_as_action_prompt(function_description: Sequence[dict]) -> str:
    prompt = "Here are the signatures and docstrings of these functions:\n\n```python\n"
    prompt += generate_function_signatures(function_description)
    prompt += "\n```"
    prompt += WRITE_CODE_PROMPT
    return prompt


_python_comment_pattern = re.compile(r"#.*")


def parse_python_code_comments(code: str) -> str:
    comments = _python_comment_pattern.findall(code)
    comments = [comment.strip() for comment in comments]
    comments = [comment if comment else "\n" for comment in comments]
    return " ".join(comments)


def extract_python_code_blocks(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]
    return text


class BaseAdapter:
    conversation_start_dict: dict[
        ActionFormat, tuple[ConversationMessage, ConversationMessage]
    ]

    @staticmethod
    def parse_react(text: str) -> ActionWithTought:
        invalid_format = False
        split_text = text.rsplit("Action:", 1)
        if len(split_text) == 0:
            thought_text, action_text = text, ""
            invalid_format = True
        elif len(split_text) == 1:
            if "search[" in text or "click[" in text:
                thought_text, action_text = "", split_text[0]
            else:
                thought_text, action_text = split_text[0], ""
            invalid_format = True
        else:
            thought_text, action_text = split_text

        thought_parts = thought_text.split("Thought:")
        if len(thought_parts) == 1:
            thought = thought_parts[0]
            invalid_format = True
        else:
            thought = thought_parts[1].strip()
        action = action_text.strip()
        if invalid_format:
            print("The text is not in the correct format. Parsing result may not be accurate.")
            print("###RAW TEXT:\n", text)
            print("\n###PARSED THOUGHT:\n", thought)
            print("\n###PARSED ACTION:\n", action)
        return ActionWithTought(thought, action)

    @staticmethod
    def to_react(action_with_thought: ActionWithTought) -> str:
        return (
            f"Thought:\n{action_with_thought.thought}\n\n"
            f"Action:\n{action_with_thought.action}"
        )

    @staticmethod
    def parse_function_calling(text: str) -> ActionWithTought:
        raise NotImplementedError

    @staticmethod
    def to_function_calling(action_with_thought: ActionWithTought) -> str:
        raise NotImplementedError

    @staticmethod
    def parse_code_as_action(text: str) -> ActionWithTought:
        raise NotImplementedError

    @staticmethod
    def to_code_as_action(action_with_thought: ActionWithTought) -> str:
        raise NotImplementedError

    @classmethod
    def action_parser(cls, action: str, action_format: ActionFormat) -> str:
        if action_format == ActionFormat.REACT:
            return cls.parse_react(action).action
        if action_format == ActionFormat.FUNCTION_CALLING:
            return cls.parse_function_calling(action).action
        if action_format == ActionFormat.CODE_AS_ACTION:
            return cls.parse_code_as_action(action).action
        raise NotImplementedError
