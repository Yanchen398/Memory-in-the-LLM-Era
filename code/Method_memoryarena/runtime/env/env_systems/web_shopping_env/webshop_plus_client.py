"""
WebshopPlus Task - Multi-step Sequential Purchase Evaluation

This module implements a complex multi-step shopping task where agents need to:
1. Purchase multiple products sequentially
2. Satisfy individual product constraints (price, attributes, options)
3. Check product compatibility across steps
4. Meet global budget constraints

Key Design:
- Each purchase in WebShop normally ends the episode (done=True)
- For multi-item tasks, we intercept the done signal and reset for the next purchase
- We track all purchases and return a summary of what was bought
- This client does not calculate rewards
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Dict, List

import requests

from .runtime.controller import (
    BaseAdapter,
    BaseEnvClient,
    BaseTask,
    extract_python_code_blocks,
    format_code_as_action_prompt,
    format_function_call_prompt,
    parse_python_code_comments,
)
from .runtime.controller.types import (
    ActionFormat,
    ActionWithTought,
    ConversationMessage,
    StepOutput,
)
from .runtime.reward_helpers import (
    CatalogLookup,
    compute_reward_for_step,
    load_catalog,
    select_catalog_files,
)
from .runtime.runtime_paths import (
    default_domain_data_path,
    default_product_catalog_dir,
)

# Collected performance timings for purchase handling.
PURCHASE_TIMINGS: list[Dict[str, float]] = []

# ============================================================================
# WebShop Function Descriptions
# ============================================================================

WEBSHOP_FUNCTION_DESCRIPTION = [
    {
        "name": "search",
        "description": "If the search bar is on the page, you can use this function to search for a product. If the action is not valid, perform nothing.",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "Keywords in search are up to you. Remember that your keywords in search should be carefully designed.",
                }
            },
            "required": ["keywords"],
        },
    },
    {
        "name": "click",
        "description": "Click on a button.",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {
                    "type": "string",
                    "description": "The item to click. The item should be one of the cilickable values on the page.",
                }
            },
            "required": ["item"],
        },
    },
]


# ============================================================================
# WebShop Adapter - Converts between different action formats
# ============================================================================

# Important instructions shown at the start of each task
IMPORTANT_INSTRUCTIONS = """IMPORTANT INSTRUCTIONS:

Available Actions (use EXACTLY ONE action per turn):
- search[keywords] - Search for products with given keywords
- click[product_id] - Click on a product (use product ID, NOT product name)
- click[Back to Search] - Return to search page from product detail page
- click[< Prev] or click[Next >] - Navigate between search result pages
- click[Buy Now] - Purchase the current product
- click[option_value] - Select product options (e.g., color, size)

CRITICAL RULES:
1. You MUST use EXACTLY ONE action per turn. You can only use the Available Actions shown on current Environment.
2. Actions MUST follow the exact format shown above (e.g., search[laptop], click[product ID])
3. Do NOT combine multiple actions in one turn

"""

# - click[Description] - View product description
# - click[Features] - View product features
# - click[Reviews] - View product reviews

class WebshopAdapter(BaseAdapter):
    conversation_start_dict = {
        ActionFormat.REACT: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": f"You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nKeywords in search are up to you, but the value in click must be a value in the list of available actions.\nRemember that your keywords in search should be carefully designed.\nYour response MUST be exactly one action in the format click[...] or search[...].\nDo NOT include thoughts or explanations.\n\n\n {IMPORTANT_INSTRUCTIONS}",
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
        ActionFormat.FUNCTION_CALLING: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": f"You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nIMPORTANT: When clicking on a product, you must click the product ID (e.g., B07XYZ123), not the product name.\nAn action should be done by invoking a function.\n\n{format_function_call_prompt(WEBSHOP_FUNCTION_DESCRIPTION)}\n\n\nIf the page remains unchanged, it might indicate that your action is invalid.",
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
        ActionFormat.CODE_AS_ACTION: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": f"You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nIMPORTANT: When clicking on a product, you must click the product ID (e.g., B07XYZ123), not the product name.\nYou can perform one of these actions by writing python code to invoke a function.\n\n{format_code_as_action_prompt(WEBSHOP_FUNCTION_DESCRIPTION)}\n\n\nIf the page remains unchanged, it might indicate that your action is invalid.",
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
    }

    @staticmethod
    def parse_react(text: str) -> ActionWithTought:
        """
        ReAct format:
        ```
        Thought:
        I think ...

        Action:
        click[something]
        ```
        """
        invalid_format_flg = False
        _split = text.rsplit("Action:", 1)
        if len(_split) == 0:
            _thought, _action = text
            invalid_format_flg = True
        elif len(_split) == 1:
            if "search[" in text or "click[" in text:
                _thought, _action = "", _split[0]
                invalid_format_flg = False
            else:
                _thought, _action = _split[0], ""
                invalid_format_flg = True
        else:
            assert len(_split) == 2
            _thought, _action = _split

        thought = _thought.split("Thought:")
        if len(thought) == 1:
            thought = thought[0]
            if not ("search[" in _action or "click[" in _action):
                invalid_format_flg = True
        else:
            thought = thought[1].strip()
        action = _action.strip()
        if invalid_format_flg:
            print(
                "The text is not in the correct format. Parsing result may not be accurate."
            )
            print("###RAW TEXT:\n", text)
            print("\n###PARSED THOUGHT:\n", thought)
            print("\n###PARSED ACTION:\n", action)
        return ActionWithTought(thought, action)

    @staticmethod
    def to_react(action_with_thought: ActionWithTought) -> str:
        return f"Thought:\n{action_with_thought.thought}\n\nAction:\n{action_with_thought.action}"

    @staticmethod
    def parse_function_calling(text: str) -> ActionWithTought:
        """
        Function Calling format:
        ```json
        {
            "thought": "I think ...",
            "function_name": "function_name",
            "arguments": {"kwarg1": "value1", "kwarg2": "value2"}
        }
        ```
        """
        _fn_call = json.loads(
            "{" + text.split("{", 1)[-1].rsplit("}", 1)[0] + "}", strict=False
        )
        thought = _fn_call["thought"]
        fn_name = _fn_call["function_name"]
        args = _fn_call["arguments"]
        if fn_name not in ["search", "click"]:
            raise ValueError("Invalid function name.")
        if fn_name == "search":
            action = f"search[{args['keywords']}]"
        else:
            action = f"click[{args['item']}]"
        return ActionWithTought(thought=thought, action=action)

    @staticmethod
    def to_function_calling(action_with_thought: ActionWithTought) -> str:
        if action_with_thought.action.startswith("search"):
            fn_name = "search"
            args = {"keywords": action_with_thought.action.split("[")[-1].split("]")[0]}
        elif action_with_thought.action.startswith("click"):
            fn_name = "click"
            args = {"item": action_with_thought.action.split("[")[-1].split("]")[0]}
        else:
            raise ValueError("Invalid action.")
        return json.dumps(
            {
                "thought": action_with_thought.thought,
                "function_name": fn_name,
                "arguments": args,
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def parse_code_as_action(text: str) -> ActionWithTought:
        def search(keywords: str):
            return f"search[{keywords}]"

        def click(item: str):
            return f"click[{item}]"

        text = extract_python_code_blocks(text)
        try:
            action = eval(text, {}, {"search": search, "click": click})
        except Exception as e:
            raise ValueError("Invalid action.") from e
        thought = parse_python_code_comments(text)
        return ActionWithTought(thought=thought, action=action)

    @staticmethod
    def to_code_as_action(action_with_thought: ActionWithTought) -> str:
        text = f"```python\n# {action_with_thought.thought}\n"
        if action_with_thought.action.startswith("search"):
            text += f"search({repr(action_with_thought.action.split('[')[-1].split(']')[0])})"
        elif action_with_thought.action.startswith("click"):
            text += f"click({repr(action_with_thought.action.split('[')[-1].split(']')[0])})"
        text += "\n```"
        return text


# ============================================================================
# Base WebShop Environment Client
# ============================================================================

class WebshopEnvClient(BaseEnvClient):
    """Base environment client for WebShop tasks."""

    adapter_cls = WebshopAdapter

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(
            f"{self.env_server_base}/create",
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise requests.RequestException(f"Failed to create environment: {ok}")
        self.conversation_start = self.adapter_cls.conversation_start_dict[
            self.action_format
        ]
        self.env_id = ok.json()

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["env_idx"] = self.env_id
        max_retries = 5
        last_response = None
        for attempt in range(max_retries):
            res = requests.post(
                f"{self.env_server_base}/{path}",
                json=data,
                timeout=self.timeout,
            )
            last_response = res
            if res.status_code == 503:
                import time

                time.sleep(0.1)
            elif res.status_code == 200:
                break
            else:
                print("---------------------")
                print(res.status_code)
                print(data)
        if res.status_code != 200:
            body_preview = ""
            if last_response is not None:
                try:
                    body_preview = last_response.text[:500]
                except Exception:
                    body_preview = "<unavailable>"
            raise requests.RequestException(
                f"Env POST {path} failed with status {res.status_code}. "
                f"Response: {body_preview}"
            )
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        if res.status_code != 200:
            body_preview = ""
            try:
                body_preview = res.text[:500]
            except Exception:
                body_preview = "<unavailable>"
            raise requests.RequestException(
                f"Env GET {path} failed with status {res.status_code}. "
                f"Response: {body_preview}"
            )
        return res.json()

    def observe(self) -> dict[str, Any]:
        response = self._get("observation")
        return response

    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        try:
            action = WebshopAdapter.action_parser(action, self.action_format)
            print(action)
        except Exception as e:
            print(e, action)
            return StepOutput(
                state="Invalid Action.\n\n" + self.observe(), reward=0.0, done=False
            )
        response = self._post("step", {"action": action})
        return StepOutput(
            state=response["state"],
            reward=0.0,  # Rewards are not used in this no-reward variant
            done=response["done"],
        )

    def reset(self, idx: int) -> dict[str, Any]:
        response = self._post("reset", {"session_id": idx})
        response[0] = self.observe()
        return response

    def close(self):
        response = self._post("close", {})


# ============================================================================
# WebshopPlus Environment Client
# ============================================================================


class WebshopPlusEnvClient(WebshopEnvClient):
    """
    Environment client for complex multi-step shopping tasks.

    Extends WebshopEnvClient to handle:
    - Task definition loading from JSON
    - Multi-step purchase tracking
    - Purchase summaries without reward calculation
    - Compatibility and constraint validation
    - Environment reset between purchases
    """

    def __init__(
        self,
        env_server_base: str,
        task_file: str,
        data_len: int,
        *args,
        timeout: int = 300,
        max_rounds: Optional[int] = None,
        product_catalog_dir: Optional[str] = None,
        domain_data_path: Optional[str] = None,
        enable_feedback: bool = False,
        **kwargs
    ):
        """
        Initialize WebshopPlusEnvClient.

        Args:
            env_server_base: Base URL of the environment server
            task_file: Path to the JSON task definition file
            data_len: Total number of data instances
            timeout: Request timeout in seconds
        """
        super().__init__(env_server_base, data_len, *args, timeout=timeout, **kwargs)
        self._init_task_state(task_file, max_rounds, enable_feedback, product_catalog_dir, domain_data_path)

    def _init_task_state(
        self,
        task_file: str,
        max_rounds: Optional[int],
        enable_feedback: bool,
        product_catalog_dir: Optional[str],
        domain_data_path: Optional[str],
    ) -> None:
        """Initialize task-specific state. Shared by __init__ and ReusableWebshopPlusEnvClient."""
        self.task_file = task_file
        self.task_def = self._load_task_definition(task_file)
        self.max_rounds = max_rounds
        self.enable_feedback = enable_feedback
        if self.max_rounds is not None and self.max_rounds > 60:
            self.no_purchase_exit_threshold = math.ceil(self.max_rounds / 3)
        else:
            self.no_purchase_exit_threshold = None
        self.product_catalog_dir = self._resolve_repo_path(product_catalog_dir, default_product_catalog_dir())
        self.domain_data_path = self._resolve_repo_path(domain_data_path, default_domain_data_path())
        self._catalog = None
        self._ordered_steps = None
        self._reward_helpers = None
        self.reset_tracking()

    def _load_task_definition(self, task_file: str) -> Dict:
        """Load task definition from JSON file."""
        with open(task_file, 'r', encoding='utf-8') as f:
            task_def = json.load(f)
        print(f"\nLoaded task: {task_def['task_id']}")
        print(f"Description: {task_def['task_description']}")
        print(f"Steps: {len(task_def['steps'])}\n")
        return task_def

    def _resolve_repo_path(self, value: Optional[str], default: Path) -> Path:
        if value:
            path = Path(value)
            return path if path.is_absolute() else path.resolve()
        return default

    def reset_tracking(self):
        """Reset all tracking variables for a new episode."""
        if not hasattr(self, "max_rounds"):
            self.max_rounds = None
        if not hasattr(self, "no_purchase_exit_threshold"):
            if self.max_rounds is not None and self.max_rounds > 60:
                self.no_purchase_exit_threshold = math.ceil(self.max_rounds / 3)
            else:
                self.no_purchase_exit_threshold = None
        if not hasattr(self, "product_catalog_dir"):
            self.product_catalog_dir = self._resolve_repo_path(
                None,
                default_product_catalog_dir(),
            )
        if not hasattr(self, "domain_data_path"):
            self.domain_data_path = self._resolve_repo_path(
                None,
                default_domain_data_path(),
            )
        if not hasattr(self, "_catalog"):
            self._catalog = None
        if not hasattr(self, "_ordered_steps"):
            self._ordered_steps = None
        if not hasattr(self, "_reward_helpers"):
            self._reward_helpers = None
        if not hasattr(self, "enable_feedback"):
            self.enable_feedback = False
        self.completed_steps = []  # List of completed step numbers
        self.purchased_asins = []  # List of purchased ASINs
        self.purchased_prices = []  # List of prices for each purchase
        self.purchased_product_texts = []  # Observations captured at purchase time for attribute checks
        self.current_total_expense = 0.0
        self.episode_done = False
        self.success = False
        self.last_purchased_asin = None  # Track last CONFIRMED purchased ASIN
        self.last_purchased_price = None  # Track last CONFIRMED purchased price
        self.current_viewing_price = None  # Track price of currently viewing product (before purchase)
        self.custom_instruction = None  # Custom instruction to override environment's instruction
        self.agent_outputs = []  # Track agent outputs for soft exit reporting
        self.last_action = None  # Last parsed action_only string
        self.same_action_count = 0  # Count consecutive identical action_only outputs
        self.max_same_action = 7  # Soft exit threshold for repeated actions
        self.round_count = 0  # Count total rounds in this episode

    def reset(self, idx: int) -> Dict[str, Any]:
        """
        Reset environment and tracking for a new episode.

        Args:
            idx: Episode index (not used for combo tasks, we use the task definition)
        """
        # Reset tracking variables
        self.reset_tracking()

        # Call parent reset (this will call the environment server)
        response = self._post("reset", {"session_id": 0})

        # Get initial observation with task instructions
        self.custom_instruction = self._create_task_instruction()
        response[0] = self.observe()

        return response

    def _create_task_instruction(self) -> str:
        """
        Create task instruction from task definition.

        Returns:
            Task instruction
        """
        return self.task_def.get('agent_instruction', '')

    def _replace_instruction_in_state(self, state: str) -> str:
        """
        Replace the environment's instruction with custom_instruction in any state text.

        This is a helper method used by both observe() and step() to ensure
        consistent instruction replacement across all states.

        Args:
            state: The state text (observation or step output state)

        Returns:
            State text with instruction replaced if custom_instruction is set
        """
        if not self.custom_instruction:
            return state

        # Find and replace instruction using regex-like approach
        # The instruction can appear in different formats:
        # 1. "WebShop [SEP] Instruction: [SEP] <text> [SEP] ..."
        # 2. "Instruction: [SEP] <text> [SEP] ..."

        if 'Instruction:' in state:
            # Split by [SEP] to find instruction part
            parts = state.split('[SEP]')

            # Find the index of "Instruction:" part
            instruction_idx = -1
            for i, part in enumerate(parts):
                if 'Instruction:' in part:
                    instruction_idx = i
                    break

            if instruction_idx >= 0 and instruction_idx + 1 < len(parts):
                # Replace the instruction content (next part after "Instruction:")
                parts[instruction_idx + 1] = f" {self.custom_instruction} "
                state = '[SEP]'.join(parts)
            else:
                # Fallback: couldn't find proper format
                print(f"[WebshopPlus] Warning: Could not replace instruction in state")
        else:
            # No instruction found, prepend our custom instruction
            state = f"WebShop [SEP] Instruction: [SEP] {self.custom_instruction} [SEP] {state}"

        return state

    def observe(self) -> str:
        """
        Get current observation from environment.

        If custom_instruction is set, replace the environment's instruction with it.
        """
        obs = super().observe()
        return self._replace_instruction_in_state(obs)

    def step(self, action: str) -> StepOutput:
        """
        Execute action in the environment.

        Key logic:
        1. Execute action in base environment
        2. Check if purchase completed
        3. Record purchase info (no immediate reward calculation)
        4. Auto-complete once required number of items have been purchased
        5. Reset environment for next action when more purchases are needed
        6. Rewards are always 0 in this no-reward variant

        Returns:
            StepOutput with state and done flag (reward is always 0.0)
        """
        # Check if combo episode already done
        if self.episode_done:
            return StepOutput(
                state="Task already completed.",
                reward=0.0,
                done=True
            )

        # Extract action part from full response (handles "Thought:\n...\nAction:\n..." format)
        action_only = self._extract_action_from_response(action)
        self._record_agent_output(action, action_only)
        self._update_repeat_action(action_only)

        # Before executing action, check current state for price info
        # This helps us track the price when viewing product pages
        current_obs = self.observe()
        if self.same_action_count >= self.max_same_action:
            return self._handle_soft_exit(
                current_obs,
                reason=f"repeat_action>={self.max_same_action}",
                detail_lines=[
                    f"Repeat Action: {action_only}",
                    f"Repeat Count: {self.same_action_count}",
                ],
            )
        self._track_price_from_obs(current_obs)

        # Execute action in base environment
        base_output = super().step(action)
        step_output = StepOutput(
            state=base_output.state,
            reward=0.0,  # No reward returned in this variant
            done=base_output.done,
        )
        self.round_count += 1

        # Apply custom instruction replacement to the returned state
        # This ensures all states show the correct instruction
        if self.custom_instruction and not self.episode_done:
            step_output = StepOutput(
                state=self._replace_instruction_in_state(step_output.state),
                reward=0.0,
                done=step_output.done
            )

        # Check if base environment indicates purchase completion
        # WebShop returns done=True when a purchase is made, but the state text
        # may not always include a thank-you string, so rely primarily on the
        # done flag.
        is_purchase = step_output.done or ("Thank you for shopping" in step_output.state or "Purchased" in step_output.state)

        if (
            not is_purchase
            and self.no_purchase_exit_threshold is not None
            and len(self.task_def.get('steps', [])) > 1
            and not self.purchased_asins
            and self.round_count >= self.no_purchase_exit_threshold
        ):
            return self._handle_no_purchase_exit(step_output.state)

        if is_purchase:
            print(f"\n[WebshopPlus] Purchase detected!")
            purchase_start = time.perf_counter()

            # Extract purchase information from environment session
            purchase_info = self._extract_purchase_info_from_env()
            extract_elapsed = time.perf_counter() - purchase_start
            print(f"[WebshopPlus] Purchase info extraction took {extract_elapsed:.3f}s")

            if purchase_info:
                asin = purchase_info['asin']
                price = purchase_info['price']
                print(f"[WebshopPlus] Purchased ASIN: {asin}, Price: ${price:.2f}")

                # Record purchase (no reward calculation yet)
                self.purchased_asins.append(asin)
                self.purchased_prices.append(price)
                self.purchased_product_texts.append(current_obs)
                self.current_total_expense += price

                # Clear viewing price
                self.current_viewing_price = None

                purchase_count = len(self.purchased_asins)
                required_purchases = len(self.task_def.get('steps', []))
                feedback_block = None
                feedback_elapsed = 0.0
                if self.enable_feedback:
                    feedback_start = time.perf_counter()
                    feedback_block = self._build_purchase_feedback(
                        raw_action_output=action,
                        purchase_index=purchase_count,
                        purchased_asin=asin,
                        purchased_price=price,
                    )
                    feedback_elapsed = time.perf_counter() - feedback_start
                    print(f"[WebshopPlus] Feedback build took {feedback_elapsed:.3f}s")

                # Auto-complete when all required items have been purchased
                if purchase_count >= required_purchases:
                    PURCHASE_TIMINGS.append(
                        {
                            "purchase_index": purchase_count,
                            "extract_info_s": round(extract_elapsed, 6),
                            "feedback_s": round(feedback_elapsed, 6),
                            "reset_observe_s": None,
                        }
                    )
                    print(f"[WebshopPlus] Required purchases reached ({purchase_count}/{required_purchases}). Auto-completing task.")
                    return self._handle_task_completion(feedback_block if self.enable_feedback else None)

                # Update custom instruction to brief version (no need to show full instructions again)
                self.custom_instruction = self._create_task_instruction()

                # Reset environment for next purchase
                reset_start = time.perf_counter()
                self._post("reset", {"session_id": 0})
                next_obs = self.observe()
                reset_elapsed = time.perf_counter() - reset_start
                print(f"[WebshopPlus] Reset+observe took {reset_elapsed:.3f}s")

                PURCHASE_TIMINGS.append(
                    {
                        "purchase_index": purchase_count,
                        "extract_info_s": round(extract_elapsed, 6),
                        "feedback_s": round(feedback_elapsed, 6),
                        "reset_observe_s": round(reset_elapsed, 6),
                    }
                )

                remaining = required_purchases - purchase_count
                item_word = "item" if remaining == 1 else "items"

                # Create continuation message
                feedback_prefix = f"{feedback_block}\n\n" if self.enable_feedback and feedback_block else ""
                continuation_msg = (
                    f"{feedback_prefix}Purchase #{purchase_count} completed. {remaining} required {item_word} remaining. "
                    f"Continue shopping to complete the task.\n\n"
                    f"{next_obs}"
                )

                return StepOutput(
                    state=continuation_msg,
                    reward=0.0,  # No reward in this variant
                    done=False
                )
            else:
                # Could not extract purchase info - error
                print("[WebshopPlus] Error: Could not extract purchase information")
                return StepOutput(
                    state=step_output.state + "\n\n[Error: Could not extract purchase information]",
                    reward=0.0,
                    done=False
                )
        else:
            # No purchase yet, continue normally
            return step_output

    def _extract_action_from_response(self, text: str) -> str:
        """
        Extract action from Agent response that may contain "Thought:" and "Action:" format.

        Args:
            text: Full agent response text

        Returns:
            Extracted action string, or original text if no "Action:" found
        """
        # Try to split by "Action:" to extract action part
        if "Action:" in text or "action:" in text:
            # Case insensitive split
            parts = re.split(r'(?i)action:', text, maxsplit=1)
            if len(parts) == 2:
                action = parts[1].strip()
                return action

        # If no "Action:" found, return original text (might be already just the action)
        return text.strip()

    def _record_agent_output(self, raw_text: str, action_only: str) -> None:
        self.agent_outputs.append(
            {
                "index": len(self.agent_outputs) + 1,
                "raw_output": raw_text,
                "action_only": action_only,
            }
        )

    def _update_repeat_action(self, action_only: str) -> None:
        if action_only == self.last_action:
            self.same_action_count += 1
        else:
            self.last_action = action_only
            self.same_action_count = 1

    def _handle_soft_exit(
        self,
        current_obs: str,
        reason: str,
        detail_lines: Optional[List[str]] = None,
    ) -> StepOutput:
        self.episode_done = True
        header_lines = [
            "=" * 60,
            "WEBSHOPPLUS SOFT EXIT",
            "=" * 60,
            f"Reason: {reason}",
        ]
        if detail_lines:
            header_lines.extend(detail_lines)
        header_lines.append("=" * 60)
        output_block = self._format_agent_outputs_block()
        state = f"{current_obs}\n\n" + "\n".join(header_lines) + "\n\n" + output_block
        return StepOutput(
            state=state,
            reward=0.0,
            done=True,
        )

    def _format_agent_outputs_block(self) -> str:
        lines = [
            "=" * 60,
            "AGENT OUTPUTS",
            "=" * 60,
            f"Total Outputs: {len(self.agent_outputs)}",
        ]
        if self.agent_outputs:
            last = self.agent_outputs[-1]
            lines.append(f"Last Action Only: {last['action_only']}")
        lines.append("")
        lines.append("Agent Outputs (full):")
        for entry in self.agent_outputs:
            lines.append(f"- Output #{entry['index']}")
            lines.append(f"  Action Only: {entry['action_only']}")
            lines.append("  Raw Output (prefixed):")
            lines.append(self._indent_text(entry["raw_output"]))
        return "\n".join(lines)

    def _handle_no_purchase_exit(self, current_state: str) -> StepOutput:
        return self._handle_soft_exit(
            current_state,
            reason="no_purchase_before_one_third",
            detail_lines=[
                f"Max Rounds: {self.max_rounds}",
                f"Threshold (ceil/3): {self.no_purchase_exit_threshold}",
                f"Rounds Elapsed: {self.round_count}",
                f"Required Purchases: {len(self.task_def.get('steps', []))}",
            ],
        )

    def _indent_text(self, text: str, prefix: str = "    OUTPUT: ") -> str:
        if text is None:
            return f"{prefix}(none)"
        lines = str(text).splitlines()
        if not lines:
            return f"{prefix}(empty)"
        return "\n".join(f"{prefix}{line}" for line in lines)

    def _load_reward_helpers(self) -> None:
        if self._reward_helpers is not None:
            return
        self._reward_helpers = {
            "CatalogLookup": CatalogLookup,
            "compute_reward_for_step": compute_reward_for_step,
            "load_catalog": load_catalog,
            "select_catalog_files": select_catalog_files,
        }

    def _get_ordered_steps(self) -> List[Dict[str, Any]]:
        ordered_steps = getattr(self, "_ordered_steps", None)
        if ordered_steps is not None:
            return self._ordered_steps
        steps = list(self.task_def.get("steps", []))
        step_nums = [step.get("step") for step in steps]
        if steps and all(isinstance(num, int) for num in step_nums):
            steps = sorted(steps, key=lambda step: step.get("step", 0))
        self._ordered_steps = steps
        return steps

    def _empty_catalog(self):
        self._load_reward_helpers()
        if self._reward_helpers and "CatalogLookup" in self._reward_helpers:
            return self._reward_helpers["CatalogLookup"](
                category_by_asin={}, name_by_asin={}
            )

        class _FallbackCatalog:
            category_by_asin: Dict[str, str] = {}
            name_by_asin: Dict[str, str] = {}

        return _FallbackCatalog()

    def _get_catalog(self):
        catalog = getattr(self, "_catalog", None)
        if catalog is not None:
            return self._catalog
        self._load_reward_helpers()
        if not self._reward_helpers:
            self._catalog = self._empty_catalog()
            return self._catalog
        load_catalog = self._reward_helpers["load_catalog"]
        select_catalog_files = self._reward_helpers["select_catalog_files"]

        catalog_files = None
        if self.product_catalog_dir.exists():
            try:
                catalog_files = select_catalog_files(
                    self.product_catalog_dir,
                    self.domain_data_path,
                    [self.task_file],
                )
            except Exception as exc:
                print(f"[WebshopPlus] Warning: catalog selection failed: {exc}")
        try:
            self._catalog = load_catalog(self.product_catalog_dir, catalog_files)
        except Exception as exc:
            print(f"[WebshopPlus] Warning: catalog load failed: {exc}")
            self._catalog = self._empty_catalog()
        return self._catalog

    def _lookup_target_name(self, target_asin: Optional[str]) -> Optional[str]:
        if not target_asin:
            return None
        catalog = self._get_catalog()
        return catalog.name_by_asin.get(str(target_asin).upper())

    def _compute_reward_entry(
        self,
        purchase_index: int,
        purchased_asin: str,
        purchased_price: float,
    ) -> Optional[Dict[str, Any]]:
        steps = self._get_ordered_steps()
        if purchase_index <= 0 or purchase_index > len(steps):
            return None
        gt_step = steps[purchase_index - 1]
        self._load_reward_helpers()
        if not self._reward_helpers:
            return None
        catalog = self._get_catalog()
        step_result = {
            "step": gt_step.get("step"),
            "target_asin": gt_step.get("target_asin"),
            "purchased_asin": purchased_asin,
            "purchased_price": purchased_price,
        }
        compute_reward_for_step = self._reward_helpers["compute_reward_for_step"]
        try:
            return compute_reward_for_step(step_result, gt_step, catalog)
        except Exception as exc:
            print(f"[WebshopPlus] Warning: reward compute failed: {exc}")
            return None

    def _format_ground_truth(
        self, target_name: Optional[str], target_asin: Optional[str]
    ) -> str:
        name = target_name or "Unknown item"
        asin = target_asin or "N/A"
        return f"{name} (ASIN: {asin})"

    def _build_purchase_feedback(
        self,
        raw_action_output: str,
        purchase_index: int,
        purchased_asin: str,
        purchased_price: float,
    ) -> str:
        reward_entry = self._compute_reward_entry(
            purchase_index=purchase_index,
            purchased_asin=purchased_asin,
            purchased_price=purchased_price,
        )
        if reward_entry:
            reward = reward_entry.get("reward", 0.0)
            target_asin = reward_entry.get("target_asin")
            target_name = reward_entry.get("target_name")
        else:
            steps = self._get_ordered_steps()
            gt_step = steps[purchase_index - 1] if purchase_index <= len(steps) else {}
            target_asin = gt_step.get("target_asin")
            target_name = self._lookup_target_name(target_asin)
            reward = 0.0

        is_correct = reward == 1.0
        ground_truth = self._format_ground_truth(target_name, target_asin)
        verdict = "CORRECT" if is_correct else "INCORRECT"
        return (
            f"The agent's answer: {raw_action_output}\n"
            f"The answer is {verdict} because the Ground Truth answer is: {ground_truth}"
        )

    def _handle_task_completion(self, feedback_block: Optional[str] = None) -> StepOutput:
        """
        Handle task completion when required purchases are complete.

        This variant only compiles a purchase summary and does not compute rewards.
        """
        self.episode_done = True

        print(f"\n[WebshopPlus] Task completed with {len(self.purchased_asins)} purchases")
        print(f"[WebshopPlus] Total expense: ${self.current_total_expense:.2f}")

        final_msg = self._create_purchase_summary()
        if feedback_block:
            final_msg = f"{feedback_block}\n\n{final_msg}"

        return StepOutput(
            state=final_msg,
            reward=0.0,
            done=True
        )

    def _track_price_from_obs(self, obs: str):
        """
        Track price information from observation.

        When viewing product pages, extract and save the price temporarily
        so we can use it during purchase confirmation if needed.

        IMPORTANT: This only updates current_viewing_price (temporary),
        NOT last_purchased_price (which only updates after confirmed purchase).
        """
        try:
            # Look for price pattern in observation
            # Format: "Price: $1949.99" or "$1949.99"
            price_patterns = [
                r'Price:\s*\$\s*([\d,]+\.?\d*)',
                r'\$\s*([\d,]+\.?\d*)',
            ]

            for pattern in price_patterns:
                match = re.search(pattern, obs)
                if match:
                    try:
                        price_str = match.group(1).replace(',', '')
                        price = float(price_str)
                        # Only update if price is reasonable (> $1 and < $100000)
                        if 1.0 < price < 100000.0:
                            self.current_viewing_price = price  # Track viewing price (not purchased price!)
                            # print(f"[WebshopPlus] Viewing product with price: ${price:.2f}")
                            break
                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            pass  # Silently ignore tracking errors

    def _extract_purchase_info_from_env(self) -> Optional[Dict]:
        """
        Extract purchase information from environment session.

        This method queries the environment server to get the current session state
        which contains the purchased ASIN and options.

        Returns:
            Dict with 'asin', 'price', 'options' or None if extraction fails
        """
        try:
            # Method 1: Get state from environment API
            state_info = self._get("state")
            url = state_info.get('url', '')
            html = state_info.get('html', '')
            asin = None
            price = None

            # Extract ASIN from URL: .../done/<asin>/<options>
            if '/done/' in url:
                parts = url.split('/done/')
                if len(parts) > 1:
                    asin_parts = parts[1].split('/')
                    if asin_parts:
                        asin = asin_parts[0]
                        print(f"[WebshopPlus] Extracted ASIN from URL: {asin}")

            # Method 2: If no ASIN from URL, try to extract from observation
            if not asin:
                obs = self.observe()
                # Format: "... [SEP] asin [SEP] B09JQSLL92 [SEP] ..."
                if '[SEP] asin [SEP]' in obs:
                    parts = obs.split('[SEP]')
                    for i, part in enumerate(parts):
                        if part.strip() == 'asin' and i + 1 < len(parts):
                            asin = parts[i + 1].strip()
                            print(f"[WebshopPlus] Extracted ASIN from observation: {asin}")
                            break

            # Extract price from HTML (multiple patterns)
            price_patterns = [
                r'Price:\s*\$?([\d,]+\.?\d*)',
                r'\$\s*([\d,]+\.?\d*)',
                r'price["\s:]+\$?([\d,]+\.?\d*)',
            ]

            for pattern in price_patterns:
                match = re.search(pattern, html, re.IGNORECASE)
                if match:
                    try:
                        price_str = match.group(1).replace(',', '')
                        price = float(price_str)
                        print(f"[WebshopPlus] Extracted price from HTML: ${price:.2f}")
                        break
                    except (ValueError, IndexError):
                        continue

            # If price extraction failed, use current viewing price (from product page before purchase)
            if price is None and self.current_viewing_price is not None:
                price = self.current_viewing_price
                print(f"[WebshopPlus] Using current viewing price: ${price:.2f}")

            if asin and price is not None:
                # Successfully extracted purchase info
                # Update last_purchased_* ONLY after confirmed purchase
                self.last_purchased_asin = asin
                self.last_purchased_price = price
                self.current_viewing_price = None  # Clear viewing price after purchase

                return {
                    'asin': asin,
                    'price': price,
                    'options': {}  # Could extract options from URL if needed
                }

            print(f"[WebshopPlus] Failed to extract purchase info (ASIN: {asin}, Price: {price})")

        except Exception as e:
            print(f"[WebshopPlus] Error extracting purchase info: {e}")

        return None

    def _create_purchase_summary(self) -> str:
        """
        Create final summary of purchases without reward computation.

        Returns:
            Formatted summary string
        """
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("WEBSHOPPLUS TASK COMPLETED")
        lines.append("=" * 60)

        lines.append(f"\nPurchases Made: {len(self.purchased_asins)}")
        lines.append(f"Required Targets: {len(self.task_def.get('steps', []))}")
        lines.append(f"Total Expense: ${self.current_total_expense:.2f}")

        budget = self.task_def.get('global_constraints', {}).get('total_expense_upper')
        if budget is not None:
            lines.append(f"Budget: ${budget:.2f}")
            lines.append(f"Under Budget: {'Yes' if self.current_total_expense <= budget else 'No'}")

        lines.append("\nPurchased Items:")
        if self.purchased_asins:
            for idx, (asin, price) in enumerate(zip(self.purchased_asins, self.purchased_prices), start=1):
                lines.append(f"  Purchase #{idx}: {asin} (${price:.2f})")
        else:
            lines.append("  None")

        # target_steps = self.task_def.get('steps', [])
        # if target_steps:
        #     lines.append("\nTarget Items:")
        #     for idx, step in enumerate(target_steps, start=1):
        #         lines.append(f"  Target {idx}: {step.get('step_description', '')}")
        #         # Hide explicit target ASINs when feedback is disabled to keep summaries lean
        #         if self.enable_feedback:
        #             lines.append(f"    Expected ASIN: {step.get('target_asin', 'N/A')}")

        lines.append("=" * 60)

        return "\n".join(lines)


class WebshopPlusTask(BaseTask):
    """
    Task class for complex multi-step shopping scenarios.

    This task evaluates agents on their ability to:
    - Plan and execute multi-step purchases
    - Manage budget constraints
    - Ensure product compatibility
    - Follow sequential ordering requirements
    """

    env_client_cls = WebshopPlusEnvClient
    env_name = "WebshopPlus"

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int = 1,
        *args,
        **kwargs,
    ):
        """
        Initialize WebshopPlusTask.

        Args:
            client_args: Arguments for environment client including:
                - env_server_base: URL of environment server
                - task_file: Path to task definition JSON
                - data_len: Number of data instances
                - timeout: Request timeout
            n_clients: Number of parallel clients (default 1)
        """
        super().__init__(client_args, n_clients, *args, **kwargs)


# Backwards compatibility with previous naming
# ComboWebshopEnvClient = WebshopPlusEnvClient
# ComboWebshopTask = WebshopPlusTask
