import os
import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .base_agent import BaseAgent

from env.env_systems.travel_planner_env.clients.base_client import BaseModelClient, ToolCall
from env.env_systems.travel_planner_env.tool_executor import ToolExecutor
from env.env_systems.travel_planner_env.tool_schemas import TOOLS
from env.env_systems.travel_planner_env.prompts import (
    AGENT_SYSTEM_PROMPT,
    AGENT_USER_PROMPT_TEMPLATE,
    HISTORY_TEMPLATE,
    BASE_PERSON_TEMPLATE,
)


def _is_terminal_context_limit_error(error: Exception) -> bool:
    """Return whether the model request cannot fit even one output token."""
    message = str(error).lower()
    markers = (
        "context length is only",
        "maximum input length",
        "context_length_exceeded",
    )
    return any(marker in message for marker in markers)


def _is_malformed_tool_arguments(error: Exception) -> bool:
    """Return whether the model emitted tool arguments that are not JSON."""
    return isinstance(error, json.JSONDecodeError)


PLAN_FIELDS = (
    "current_city",
    "transportation",
    "breakfast",
    "attraction",
    "lunch",
    "dinner",
    "accommodation",
)


def _submission_tool(expected_days: int) -> Dict:
    day_properties = {
        "day": {"type": "integer", "minimum": 1, "maximum": expected_days},
    }
    day_properties.update(
        {field: {"type": "string", "minLength": 1} for field in PLAN_FIELDS}
    )
    return {
        "type": "function",
        "function": {
            "name": "SubmitTravelPlans",
            "description": (
                "Submit complete final travel plans. Every plan must have exactly "
                f"{expected_days} unique, consecutive days and every required field."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plans": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "days": {
                                    "type": "array",
                                    "minItems": expected_days,
                                    "maxItems": expected_days,
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": day_properties,
                                        "required": ["day", *PLAN_FIELDS],
                                    },
                                },
                            },
                            "required": ["name", "days"],
                        },
                    }
                },
                "required": ["plans"],
                "additionalProperties": False,
            },
        },
    }


def _validate_plan_payload(payload: Dict, expected_days: int) -> List[Dict]:
    plans = payload.get("plans") if isinstance(payload, dict) else None
    if not isinstance(plans, list) or not plans:
        raise ValueError("SubmitTravelPlans requires a non-empty plans array")

    normalized = []
    seen_names = set()
    required_days = list(range(1, expected_days + 1))
    for plan in plans:
        if not isinstance(plan, dict):
            raise ValueError("Each submitted plan must be an object")
        name = str(plan.get("name") or "").strip()
        name_key = name.casefold()
        if not name or name_key in seen_names:
            raise ValueError(f"Missing or duplicate traveler name: {name!r}")
        seen_names.add(name_key)

        days = plan.get("days")
        if not isinstance(days, list) or len(days) != expected_days:
            count = len(days) if isinstance(days, list) else "non-list"
            raise ValueError(f"{name}: expected exactly {expected_days} days, got {count}")
        day_numbers = [day.get("day") if isinstance(day, dict) else None for day in days]
        if sorted(day_numbers) != required_days or len(set(day_numbers)) != expected_days:
            raise ValueError(f"{name}: days must be unique and consecutive: {required_days}")

        normalized_days = []
        for day in sorted(days, key=lambda item: item["day"]):
            normalized_day = {"day": int(day["day"])}
            for field in PLAN_FIELDS:
                value = str(day.get(field) or "").strip()
                if not value:
                    raise ValueError(f"{name} Day {day['day']}: empty {field}")
                normalized_day[field] = value
            normalized_days.append(normalized_day)
        normalized.append({"name": name, "days": normalized_days})
    return normalized


def _render_plans(plans: List[Dict]) -> str:
    labels = {
        "current_city": "Current City",
        "transportation": "Transportation",
        "breakfast": "Breakfast",
        "attraction": "Attraction",
        "lunch": "Lunch",
        "dinner": "Dinner",
        "accommodation": "Accommodation",
    }
    rendered = []
    for plan in plans:
        lines = [f"=== {plan['name']}'s Plan ==="]
        for day in plan["days"]:
            lines.append(f"Day {day['day']}:")
            lines.extend(f"{labels[field]}: {day[field]}" for field in PLAN_FIELDS)
            lines.append("")
        rendered.append("\n".join(lines).rstrip())
    return "\n\n".join(rendered)


def _parse_strict_text(text: str, expected_days: int) -> List[Dict]:
    if not text:
        raise ValueError("Empty final plan")
    heading = re.compile(r"===\s*([^=]+?)'s Plan\s*===")
    matches = list(heading.finditer(text))
    if not matches:
        raise ValueError("Missing traveler plan heading")

    plans = []
    key_map = {
        "current city": "current_city",
        "transportation": "transportation",
        "breakfast": "breakfast",
        "attraction": "attraction",
        "lunch": "lunch",
        "dinner": "dinner",
        "accommodation": "accommodation",
    }
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end]
        day_matches = list(re.finditer(r"Day\s*(\d+)\s*:", body, re.IGNORECASE))
        days = []
        for day_index, day_match in enumerate(day_matches):
            day_end = (
                day_matches[day_index + 1].start()
                if day_index + 1 < len(day_matches)
                else len(body)
            )
            day = {"day": int(day_match.group(1))}
            for line in body[day_match.end():day_end].splitlines():
                if ":" not in line:
                    continue
                label, value = line.split(":", 1)
                field = key_map.get(label.strip().casefold())
                if field:
                    day[field] = value.strip()
            days.append(day)
        plans.append({"name": match.group(1).strip(), "days": days})
    return _validate_plan_payload({"plans": plans}, expected_days)


@dataclass
class AgentStep:
    """Record of a single agent step"""
    step_idx: int
    thought: Optional[str]
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[List[Dict]]
    final_output: Optional[str]
    raw_response: Optional[Dict] = None


@dataclass
class AgentResult:
    """Result from running the agent for one person"""
    name: str
    query: str
    final_plan: str
    scratchpad: List[AgentStep] = field(default_factory=list)
    total_steps: int = 0
    success: bool = True
    error_message: Optional[str] = None


class TravelPlannerAgent(BaseAgent):
    """
    Travel planning agent that uses tools to gather information
    and creates travel plans.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        max_steps: int = 30,
        db_path: str = None,
        system_prompt: str = None,
        strict_plan_submission: bool = False,
        strict_submission_attempts: int = 3,
        strict_search_step_limit: int = 6,
        strict_submission_max_tokens: int = 4096,
    ):
        super().__init__(model_name, temperature)
        self.client = self._create_client(model_name, max_tokens)
        self.executor = ToolExecutor(db_path=db_path)
        self.max_steps = max_steps
        self.strict_plan_submission = strict_plan_submission
        self.strict_submission_attempts = strict_submission_attempts
        self.strict_search_step_limit = strict_search_step_limit
        self.strict_submission_max_tokens = strict_submission_max_tokens

        self.system_prompt = system_prompt or AGENT_SYSTEM_PROMPT
        self.base_messages: List[Dict] = [{"role": "system", "content": self.system_prompt}]
        self.accumulated_plans: str = ""
        self.base_name: str = ""
        self.base_query: str = ""
        self.all_queries: List[str] = []
        self.previous_judgement: str = ""

        self._last_result: Optional[AgentResult] = None

        self._pending_name: Optional[str] = None
        self._pending_round_idx: Optional[int] = None
        self._pending_include_previous_plans: bool = True
        self._pending_memory_context: Optional[str] = None
        self._pending_memory_system = None
        self._pending_expected_days: Optional[int] = None

    def _create_client(self, model_name: str, max_tokens: int) -> BaseModelClient:
        model_lower = model_name.lower()

        if "gemini" in model_lower:
            from env.env_systems.travel_planner_env.clients.gemini_client import GeminiClient
            return GeminiClient(model_name=model_name)
        elif "claude" in model_lower or "anthropic" in model_lower:
            if os.environ.get("OPENAI_API_BASE"):
                from env.env_systems.travel_planner_env.clients.openai_client import OpenAIClient
                return OpenAIClient(model_name=model_name, max_tokens=max_tokens)
            else:
                from env.env_systems.travel_planner_env.clients.anthropic_client import AnthropicClient
                return AnthropicClient(model_name=model_name)
        else:
            from env.env_systems.travel_planner_env.clients.openai_client import OpenAIClient
            return OpenAIClient(model_name=model_name, max_tokens=max_tokens)

    def set_base_person(self, name: str, query: str, plan: str):
        self.base_name = name
        self.base_query = query
        self.all_queries = [f"{name}: {query}"]

        base_context = BASE_PERSON_TEMPLATE.format(
            base_name=name,
            base_query=query,
            base_plan=plan,
        )

        self.base_messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": base_context},
        ]
        self.accumulated_plans = plan

    def add_judge_feedback(self, feedback: str):
        self.previous_judgement = feedback

    def prepare_for_person(
        self,
        name: str,
        round_idx: int,
        include_previous_plans: bool = True,
        memory_context: Optional[str] = None,
        memory_system=None,
        expected_days: Optional[int] = None,
    ):
        self._pending_name = name
        self._pending_round_idx = round_idx
        self._pending_include_previous_plans = include_previous_plans
        self._pending_memory_context = memory_context
        self._pending_memory_system = memory_system
        self._pending_expected_days = expected_days

    def act(self, prompt: str) -> str:
        name = self._pending_name or "User"
        round_idx = self._pending_round_idx or 1
        include_previous_plans = self._pending_include_previous_plans
        memory_context = self._pending_memory_context
        memory_system = self._pending_memory_system
        expected_days = self._pending_expected_days

        result = self.run_single_person(
            query=prompt,
            name=name,
            round_idx=round_idx,
            include_previous_plans=include_previous_plans,
            memory_context=memory_context,
            memory_system=memory_system,
            expected_days=expected_days,
        )
        return result.final_plan

    @property
    def last_result(self) -> Optional[AgentResult]:
        return self._last_result

    def get_scratchpad_dict(self, model_type: str = None) -> list:
        if self._last_result is None:
            return []
        return self._scratchpad_to_dict(self._last_result.scratchpad, model_type=model_type)

    def run_single_person(
        self,
        query: str,
        name: str,
        round_idx: int,
        include_previous_plans: bool = True,
        memory_context: str = None,
        memory_system=None,
        raw_query: bool = False,
        expected_days: Optional[int] = None,
    ) -> AgentResult:
        result = AgentResult(name=name, query=query, final_plan="")

        self.all_queries.append(f"{name}: {query}")

        messages = list(self.base_messages)

        if memory_context:
            messages.append({
                "role": "user",
                "content": f"Here is relevant context from memory that may help with this planning task:\n\n{memory_context}",
            })

        if round_idx > 1 and include_previous_plans:
            history_context = HISTORY_TEMPLATE.format(
                all_queries="\n".join(self.all_queries),
                previous_plan=self.accumulated_plans,
                judgement=self.previous_judgement or "",
            )
            messages.append({"role": "user", "content": history_context})

        if raw_query:
            user_message = query
        else:
            user_message = AGENT_USER_PROMPT_TEMPLATE.format(name=name, query=query)

        messages.append({"role": "user", "content": user_message})
        strict_tool_names = set()

        for step_idx in range(self.max_steps):
            step = AgentStep(
                step_idx=step_idx,
                thought=None,
                tool_calls=None,
                tool_results=None,
                final_output=None,
                raw_response=None,
            )

            try:
                response = self.client.chat_with_tools(messages, TOOLS)
            except Exception as error:
                if _is_terminal_context_limit_error(error):
                    failure_label = "Terminal context-limit failure"
                    log_message = (
                        "[Agent] Context limit reached with the full official "
                        "message history; recording this round as failed."
                    )
                elif _is_malformed_tool_arguments(error):
                    failure_label = "Malformed model tool arguments"
                    log_message = (
                        "[Agent] Model emitted malformed JSON tool arguments; "
                        "recording this round as failed."
                    )
                else:
                    raise
                if self.strict_plan_submission and expected_days:
                    result.total_steps = step_idx + 1
                    print(f"{log_message} Attempting strict final submission.")
                    break
                result.success = False
                result.error_message = (
                    f"{failure_label}: "
                    f"{type(error).__name__}: {error}"
                )
                result.total_steps = step_idx + 1
                print(log_message)
                break

            print(f"\n[DEBUG Step {step_idx}]")
            print(f"  content: {repr(response.content)[:200] if response.content else None}")
            print(f"  tool_calls: {response.tool_calls}")
            print(f"  raw_response type: {type(response.raw_response)}")

            step.thought = response.content

            if hasattr(response, 'raw_response'):
                if isinstance(response.raw_response, dict):
                    step.raw_response = response.raw_response
                else:
                    try:
                        step.raw_response = response.raw_response.to_dict() if hasattr(response.raw_response, 'to_dict') else None
                    except Exception:
                        step.raw_response = None

            if response.tool_calls:
                strict_tool_names.update(tc.name for tc in response.tool_calls)
                step.tool_calls = [
                    {"id": tc.id, "name": tc.name, "args": tc.arguments}
                    for tc in response.tool_calls
                ]

                messages.append(self.client.format_assistant_tool_calls(response.tool_calls))

                step.tool_results = []
                for tc in response.tool_calls:
                    tool_result = self.executor.execute(tc.name, tc.arguments)

                    step.tool_results.append({
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "result": tool_result,
                    })

                    messages.append(self.client.format_tool_result(tc.id, tool_result, name=tc.name))

                if memory_system:
                    wrapped = memory_system.wrap_user_prompt(query)
                    step_memory = wrapped.split("</memory_context>")[0] + "</memory_context>"
                    step_memory = step_memory.strip()
                    messages.append({
                        "role": "user",
                        "content": f"[Step Memory] Updated context from memory:\n\n{step_memory}",
                    })

            else:
                candidate = response.content or ""
                if self.strict_plan_submission and expected_days:
                    try:
                        plans = _parse_strict_text(candidate, expected_days)
                        if name.casefold() not in {
                            plan["name"].casefold() for plan in plans
                        }:
                            raise ValueError(f"Final output omitted current traveler {name}")
                        candidate = _render_plans(plans)
                    except ValueError as error:
                        print(f"[Agent] Invalid free-text final plan: {error}")
                        messages.append({"role": "assistant", "content": candidate})
                        result.scratchpad.append(step)
                        break
                step.final_output = candidate
                result.scratchpad.append(step)
                result.final_plan = candidate
                result.total_steps = step_idx + 1
                break

            result.scratchpad.append(step)
            core_search_tools = {
                "FlightSearch",
                "RestaurantSearch",
                "AccommodationSearch",
                "AttractionSearch",
            }
            if self.strict_plan_submission and expected_days and (
                core_search_tools.issubset(strict_tool_names)
                or len(result.scratchpad) >= self.strict_search_step_limit
            ):
                result.total_steps = len(result.scratchpad)
                print(
                    "[Agent] Search evidence collected; forcing strict final "
                    f"submission after {result.total_steps} tool steps with "
                    f"tools={sorted(strict_tool_names)}."
                )
                break

        else:
            result.total_steps = self.max_steps

        if self.strict_plan_submission and expected_days and not result.final_plan:
            try:
                result.final_plan, submit_step = self._force_strict_submission(
                    messages, name, expected_days
                )
                result.scratchpad.append(submit_step)
                result.total_steps = max(result.total_steps, len(result.scratchpad))
                result.success = True
                result.error_message = None
            except Exception as error:
                result.success = False
                result.error_message = (
                    "Strict final submission failed: "
                    f"{type(error).__name__}: {error}"
                )
        elif not result.final_plan and result.error_message is None:
            result.success = False
            result.error_message = f"Max steps ({self.max_steps}) reached without final output"

        if result.final_plan:
            self.accumulated_plans += f"\n\n{result.final_plan}"

        self.previous_judgement = ""

        self._last_result = result
        return result

    def _force_strict_submission(
        self,
        messages: List[Dict],
        current_name: str,
        expected_days: int,
    ) -> tuple[str, AgentStep]:
        tool = _submission_tool(expected_days)
        tool_choice = {
            "type": "function",
            "function": {"name": "SubmitTravelPlans"},
        }
        final_messages = list(messages)
        final_messages.append({
            "role": "user",
            "content": (
                "Stop searching and submit the complete final plan now using "
                "SubmitTravelPlans. Include the current traveler "
                f"{current_name!r}. Each traveler must have exactly "
                f"{expected_days} unique consecutive days and every field must "
                "be non-empty; use '-' only where the benchmark allows it."
            ),
        })
        last_error = None
        for attempt in range(1, self.strict_submission_attempts + 1):
            try:
                response = self.client.chat_with_tools(
                    final_messages,
                    [tool],
                    tool_choice=tool_choice,
                    max_tokens=self.strict_submission_max_tokens,
                )
                calls = response.tool_calls or []
                if len(calls) != 1 or calls[0].name != "SubmitTravelPlans":
                    raise ValueError("Model did not call SubmitTravelPlans exactly once")
                plans = _validate_plan_payload(calls[0].arguments, expected_days)
                if current_name.casefold() not in {
                    plan["name"].casefold() for plan in plans
                }:
                    raise ValueError(f"Submission omitted current traveler {current_name}")
                rendered = _render_plans(plans)
                raw_response = response.raw_response
                if hasattr(raw_response, "to_dict"):
                    raw_response = raw_response.to_dict()
                elif not isinstance(raw_response, dict):
                    raw_response = None
                step = AgentStep(
                    step_idx=self.max_steps + attempt - 1,
                    thought=response.content,
                    tool_calls=[{
                        "id": calls[0].id,
                        "name": calls[0].name,
                        "args": calls[0].arguments,
                    }],
                    tool_results=None,
                    final_output=rendered,
                    raw_response=raw_response,
                )
                return rendered, step
            except Exception as error:
                last_error = error
                print(
                    f"[Agent] Strict submission attempt {attempt}/"
                    f"{self.strict_submission_attempts} failed: {error}"
                )
        raise RuntimeError(str(last_error))

    def build_memory_entry(
        self,
        task: str,
        action: str,
        observation: Optional[Dict] = None,
        reward: Optional[float] = None,
    ) -> str:
        if self._last_result is None:
            return json.dumps({"task": task, "action": action}, ensure_ascii=False)

        scratchpad_dict = self._scratchpad_to_dict(self._last_result.scratchpad)
        chunk = {
            "name": self._last_result.name,
            "query": task,
            "scratchpad": scratchpad_dict,
            "final_plan": action,
        }
        judgement = None
        if observation and isinstance(observation, dict):
            judgement = observation.get("judgement")
        if judgement:
            chunk["judgement"] = judgement
        return json.dumps(chunk, ensure_ascii=False)

    def _scratchpad_to_dict(self, scratchpad: List[AgentStep], model_type: str = None) -> list:
        if model_type is None:
            model_type = "openai"
            if "gemini" in self.model_name.lower():
                model_type = "gemini"
            elif "claude" in self.model_name.lower() or "anthropic" in self.model_name.lower():
                model_type = "anthropic"

        result = []
        for step in scratchpad:
            step_dict = {
                'step_idx': step.step_idx,
                'thought': step.thought,
                'final_output': step.final_output,
            }
            if step.tool_calls:
                step_dict['tool_calls'] = step.tool_calls
            if step.tool_results:
                step_dict['tool_results'] = step.tool_results
            if model_type in ["gemini", "anthropic"]:
                if step.raw_response:
                    step_dict['raw_response'] = step.raw_response
            result.append(step_dict)
        return result

    def reset(self):
        self.base_messages = [{"role": "system", "content": self.system_prompt}]
        self.accumulated_plans = ""
        self.base_name = ""
        self.base_query = ""
        self.all_queries = []
        self.previous_judgement = ""
        self._last_result = None
        self._pending_name = None
        self._pending_round_idx = None
        self._pending_include_previous_plans = True
        self._pending_memory_context = None
        self._pending_memory_system = None
        self._pending_expected_days = None

    def get_usage_stats(self) -> Dict:
        return self.client.get_usage_stats()
