# import openai
import json

from openai import OpenAI

from memos.configs.llm import OpenAILLMConfig
from memos.llms.base import BaseLLM
from memos.llms.utils import remove_thinking_tags
from memos.log import get_logger
from memos.types import MessageList


logger = get_logger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI LLM class."""

    def __init__(self, config: OpenAILLMConfig):
        self.config = config
        # self.client = openai.Client(api_key=config.api_key, base_url=config.api_base)
        self.client = OpenAI(api_key=config.api_key, base_url=config.api_base)

    def generate(self, messages: MessageList) -> str:
        """Generate a response from OpenAI LLM."""
        model_name = str(self.config.model_name_or_path)
        if model_name.lower().startswith("gpt-5.4"):
            response_content = ""
            invalid_json = False
            incomplete = False
            output_budget = int(self.config.max_tokens)
            for attempt in range(3):
                response = self.client.responses.create(
                    model=model_name,
                    input=messages,
                    max_output_tokens=output_budget,
                )
                response_content = response.output_text or ""
                candidate = (
                    response_content.replace("```", "").replace("json", "").strip()
                )
                invalid_json = False
                if candidate.startswith(("{", "[")):
                    try:
                        json.loads(candidate)
                    except json.JSONDecodeError:
                        invalid_json = True
                incomplete = getattr(response, "status", None) == "incomplete"
                if not incomplete and not invalid_json:
                    break
                reason = "incomplete output" if incomplete else "malformed JSON"
                logger.warning(
                    f"Responses API returned {reason} for {model_name} with "
                    f"max_output_tokens={output_budget}; retrying "
                    f"({attempt + 1}/3)."
                )
                output_budget = min(output_budget * 2, 65536)
            if incomplete or invalid_json:
                raise RuntimeError(
                    "Responses API failed to return complete valid JSON after 3 attempts"
                )
        else:
            request = {
                "model": model_name,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            }
            if "qwen" in model_name.lower():
                request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            response = self.client.chat.completions.create(**request)
            response_content = response.choices[0].message.content
        logger.info(f"Response from OpenAI: {response.model_dump_json()}")
        if self.config.remove_think_prefix:
            return remove_thinking_tags(response_content)
        else:
            return response_content
