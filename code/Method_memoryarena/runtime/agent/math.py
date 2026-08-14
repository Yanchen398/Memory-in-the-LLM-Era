from typing import Any, Dict, Optional, List
import json
from .base_agent import BaseAgent
import os
import sys
import re
import math
import pdb

# Import backend system
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'env', 'env_systems'))
from formal_reasoning_env.llm_backend import create_backend, LLMBackend

try:
    from openai import BadRequestError, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Some features may be limited.")


QWEN35_CHAT_EXTRA_BODY = {
    "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": False},
}
CONTEXT_LIMIT_RE = re.compile(
    r"You passed (\d+) input tokens and requested (\d+) output tokens.*?"
    r"context length is only (\d+) tokens",
    re.IGNORECASE,
)
CONTEXT_LIMIT_TRANSFORMERS_RE = re.compile(
    r"maximum context length is (\d+) tokens.*?"
    r"requested (\d+) tokens \((\d+) in the messages, "
    r"(\d+) in the completion\)",
    re.IGNORECASE,
)
CONTEXT_RETRY_SAFETY_TOKENS = 64
CONTEXT_RETRY_MAX_ATTEMPTS = 8
CONTEXT_RETRY_MIN_TOKENS = 1
CONTEXT_RETRY_MIN_REDUCTION = 1024


def _is_qwen_model(model_name: str) -> bool:
    return "qwen" in (model_name or "").lower()


def _context_limited_max_tokens(
    error: Exception,
    requested: int,
) -> Optional[int]:
    error_text = str(error)
    match = CONTEXT_LIMIT_RE.search(error_text)
    if match:
        input_tokens, _, context_tokens = (
            int(value) for value in match.groups()
        )
    else:
        match = CONTEXT_LIMIT_TRANSFORMERS_RE.search(error_text)
        if not match:
            return None
        context_tokens, _, input_tokens, _ = (
            int(value) for value in match.groups()
        )

    raw_available = context_tokens - input_tokens
    if (
        raw_available < CONTEXT_RETRY_MIN_TOKENS
        or requested <= CONTEXT_RETRY_MIN_TOKENS
    ):
        return None
    safety_cap = max(
        CONTEXT_RETRY_MIN_TOKENS,
        raw_available - CONTEXT_RETRY_SAFETY_TOKENS,
    )
    reduction = max(CONTEXT_RETRY_MIN_REDUCTION, requested // 4)
    material_cap = max(
        CONTEXT_RETRY_MIN_TOKENS,
        requested - reduction,
    )
    retry_tokens = min(safety_cap, material_cap)
    if retry_tokens >= requested:
        return None
    return retry_tokens


def _create_completion_with_context_retry(client: Any, request: Dict[str, Any]):
    current_request = dict(request)
    for attempt in range(CONTEXT_RETRY_MAX_ATTEMPTS):
        try:
            return client.chat.completions.create(**current_request)
        except BadRequestError as error:
            if not _is_qwen_model(str(current_request.get("model", ""))):
                raise
            requested = current_request.get("max_tokens")
            if requested is None:
                raise
            retry_max_tokens = _context_limited_max_tokens(
                error,
                int(requested),
            )
            if (
                retry_max_tokens is None
                or attempt + 1 >= CONTEXT_RETRY_MAX_ATTEMPTS
            ):
                raise
            print(
                "Context boundary retry: preserving the full prompt and "
                f"reducing max_tokens from {requested} to "
                f"{retry_max_tokens} with an adaptive reduction and a "
                f"{CONTEXT_RETRY_SAFETY_TOKENS}-token safety margin "
                f"(attempt {attempt + 2}/{CONTEXT_RETRY_MAX_ATTEMPTS})."
            )
            current_request["max_tokens"] = retry_max_tokens


class MathAgent(BaseAgent):
    """Agent for Math tasks."""

    EMPTY_TEXT_FALLBACK = "No content was provided."

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        backend: str = "openai",
        base_url: Optional[str] = None,
    ):
        super().__init__(model_name, temperature)
        self.history = []
        self.max_tokens = max_tokens
        self.backend_name = backend
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        # Initialize LLM backend
        try:
            
            self.llm_backend = create_backend(
                backend_name=backend,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=self.base_url
            )
        except Exception as e:
            print(f"Warning: Failed to initialize {backend} backend: {e}")
            print("Falling back to OpenAI backend")
            if OPENAI_AVAILABLE:
                self.llm_backend = create_backend(
                    backend_name="openai",
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_url=self.base_url
                )
            else:
                raise RuntimeError("No LLM backend available")

    @classmethod
    def _non_empty_text(cls, value: Any, fallback: Optional[str] = None) -> str:
        """Return a provider-safe, non-empty text value."""
        if value is None:
            return fallback or cls.EMPTY_TEXT_FALLBACK
        text = str(value).strip()
        return text if text else (fallback or cls.EMPTY_TEXT_FALLBACK)

    @staticmethod
    def _is_empty_text_bad_request(error: Exception) -> bool:
        return (
            error.__class__.__name__ == "BadRequestError"
            and "text content blocks must be non-empty" in str(error)
        )

    @staticmethod
    def _env_int(name: str, default: int, minimum: int = 1) -> int:
        try:
            return max(minimum, int(os.getenv(name, str(default))))
        except ValueError:
            return default

    @staticmethod
    def _format_errors(errors: List[str]) -> str:
        return " | ".join(errors[-3:])

    def _empty_final_action(
        self,
        prompt: str,
        errors: Optional[List[str]] = None,
        tool_trace: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        trace = tool_trace or []
        if errors:
            trace = trace + [{
                "tool": "reasoning",
                "input": prompt,
                "result": "",
                "error": self._format_errors(errors),
            }]
        return {
            "type": "final",
            "answer": "",
            "tool_trace": trace,
            "tool_info": trace,
        }
       
    def build_prompt(
        self,
        task: str,
        background: str = None,
        memory_context: Optional[str] = None,
    ) -> str:
        """
        Build a formatted prompt for the math problem.
        If the task contains BACKGROUND and PROBLEM sections, keep them.
        Otherwise, treat the entire task as the PROBLEM.
        Optionally wraps with memory context.
        """
        # Check if task already has BACKGROUND/PROBLEM format
        if "### BACKGROUND" in task or "### PROBLEM" in task:
            formatted_task = task
        else:
            # Otherwise, create formatted prompt
            formatted_task = f"""
            ### BACKGROUND:
            {background if background else "No information provided."}
            ### PROBLEM:
            {task}"""
        
        # Add memory context if provided
        if memory_context:
            memory_prompt = f"""<memory_context>
{memory_context}
</memory_context>

"""
            return memory_prompt + formatted_task
        
        return formatted_task
    
    def build_memory_entry(
        self,
        task: str,
        action: str,
        observation: Optional[Dict[str, Any]] = None,
        reward: Optional[float] = None,
    ) -> str:
        memory_entry = f"## Task: {task}\n"
        if action['type'] == 'final':
            memory_entry += f"## solution: {action['answer']}\n"
        if reward is not None:
            memory_entry += f"## Judge: {'CORRECT' if reward else 'INCORRECT'}\n"
        if observation is not None:
            if "tool_info" in observation and len(observation["tool_info"])!=0: 
                tool_info= str(observation["tool_info"]) 
                memory_entry += f"## Tool Calls Info: {tool_info}\n"
        return memory_entry
    
    def _validate_code_safety(self, code: str) -> tuple:
        """
        Validate code for security risks.
        Returns (is_safe, error_message)
        """
        dangerous_patterns = [
            (r'\bos\.remove\b|\bos\.rmdir\b|\bshutil\.rmtree\b', "File deletion operations not allowed"),
            (r'\bopen\(|\.write\(', "File write operations not allowed"),
            (r'\bos\.system\b|\bsubprocess\b|\bpopen\b', "System command execution not allowed"),
            (r'\bexec\(|\beval\(|\bcompile\(', "Dynamic code execution not allowed"),
            (r'\b__import__\b', "Dynamic imports not allowed"),
            (r'\bos\.path\b.*remove|unlink', "Path operations not allowed"),
            (r'\bkill|signal\.SIGKILL|\bos\.kill', "Process termination not allowed"),
            (r'\bsocket\b|\burllib\b|\bhttps?\b', "Network operations not allowed"),
            (r'__', "Private attribute access not allowed"),
            (r'\bimport\s+os\b|\bfrom\s+os\b', "OS module import not allowed"),
        ]
        
        for pattern, error_msg in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Security violation: {error_msg}"
        
        if re.search(r'(rm\s+-rf|delete|drop\s+table)', code, re.IGNORECASE):
            return False, "Suspicious shell commands detected"
        
        return True, ""
    
    def _create_safe_environment(self) -> Dict[str, Any]:
        """Create a restricted environment for code execution."""
        safe_builtins = {
            'print': print,
            'len': len,
            'range': range,
            'sum': sum,
            'abs': abs,
            'min': min,
            'max': max,
            'sorted': sorted,
            'round': round,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'zip': zip,
            'enumerate': enumerate,
            'map': map,
            'filter': filter,
            'math': math,
        }
        
        try:
            import numpy as np
            safe_builtins['np'] = np
        except ImportError:
            pass
        
        try:
            import statistics
            safe_builtins['statistics'] = statistics
        except ImportError:
            pass
        
        return {'__builtins__': safe_builtins}
    
    def _reasoning_with_errors(self, task: str) -> tuple:
        """Call LLM for symbolic reasoning. Return empty result after retries fail."""
        task = self._non_empty_text(task, "Solve the original math problem.")
        prompt = f"""Solve the following math problem carefully.
Provide concise reasoning, check your work, and state the final answer clearly.

Problem: {task}
"""
        messages = [
            {"role": "system", "content": "You are a careful math solver."},
            {"role": "user", "content": prompt},
        ]
        errors = []
        for attempt in range(self._env_int("MATH_AGENT_REASONING_RETRIES", 3)):
            try:
                result = (self.llm_backend.chat(messages=messages) or "").strip()
            except Exception as e:
                errors.append(f"attempt {attempt + 1}: {type(e).__name__}: {str(e)[:200]}")
                continue
            if result:
                return result, errors
            errors.append(f"attempt {attempt + 1}: empty response")
        return "", errors

    def reasoning(self, task: str) -> str:
        """Call LLM for symbolic reasoning."""
        result, _ = self._reasoning_with_errors(task)
        return result
    
    def execute_code(self, code: str) -> str:
        """Execute Python code safely."""
        is_safe, error_msg = self._validate_code_safety(code)
        if not is_safe:
            return f"Error: {error_msg}"
        
        try:
            safe_env = self._create_safe_environment()
            output_capture = []
            
            def safe_print(*args, **kwargs):
                output_capture.append(' '.join(str(arg) for arg in args))
            
            safe_env['print'] = safe_print
            exec(code, safe_env)
            
            result_output = '\n'.join(output_capture)
            return result_output.strip() if result_output.strip() else "Code executed successfully"
            
        except Exception as e:
            error_type = type(e).__name__
            return f"Error: {error_type}: {str(e)[:100]}"
    
    def act(self, prompt: str) -> str:
        """
        Generate math solving action using LLM with function calling.
        
        Your prompt should have all info re. task and memory (retrieved info from memory system injected with observations.)
        
        Override this method to use your preferred LLM client.
        """
        # prompt = self.build_prompt(prompt)
        prompt = self._non_empty_text(prompt, "No math problem was provided.")
        
        # Tool calling is primarily supported by OpenAI
        # For other backends, we can use the reasoning tool directly
        if self.backend_name == "openai" and OPENAI_AVAILABLE:
            try:
                action = self._act_with_tools(prompt)
            except Exception as e:
                result, errors = self._reasoning_with_errors(prompt)
                errors.insert(0, f"tool loop failed: {type(e).__name__}: {str(e)[:200]}")
                if result:
                    action = {
                        "type": "final",
                        "answer": result,
                        "tool_trace": [{
                            "tool": "reasoning",
                            "input": prompt,
                            "result": result,
                            "error": self._format_errors(errors),
                        }],
                        "tool_info": [{
                            "tool": "reasoning",
                            "input": prompt,
                            "result": result,
                            "error": self._format_errors(errors),
                        }],
                    }
                else:
                    action = self._empty_final_action(prompt, errors=errors)
            action["input"] = prompt
            return action
        else:
            # For non-OpenAI backends, use reasoning directly
            result, errors = self._reasoning_with_errors(prompt)
            return {
                "type": "tool_result",
                "tool": "reasoning",
                "input": prompt,
                "result": result,
                "error": self._format_errors(errors) if errors else None,
            }
    
    def _act_with_tools(self, prompt: str) -> Dict[str, Any]:
        """Internal method for OpenAI-style tool calling."""
        from openai import OpenAI
        prompt = self._non_empty_text(prompt, "No math problem was provided.")
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "reasoning",
                    "description": "Solve a math problem using step-by-step reasoning",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string", "description": "The math problem to solve"}
                        },
                        "required": ["input"]
                    }
                }
            },
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "coding",
            #         "description": "Execute Python code to solve computational problems",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "input": {"type": "string", "description": "Python code to execute"}
            #             },
            #             "required": ["input"]
            #         }
            #     }
            # }
        ]
        
        system_prompt = """You are a mathematical reasoning assistant.

Your task is to solve the math problem described in PROBLEM using the definitions and setup in BACKGROUND if there is any.
You can also use memory context if useful and provided.

### AVAILABLE TOOLS:
1. **reasoning**: Use this for step-by-step mathematical reasoning, symbolic manipulation, self-checking, and retrying a solution path.

You may call the reasoning tool multiple times. After each tool result, reflect on whether the answer is complete and consistent with the problem. If there is an error or uncertainty, call the tool again with a more targeted request. When you are confident, return a final answer for verification.
"""
 
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=self.base_url,
        )
        max_tool_iterations = self._env_int("MATH_AGENT_MAX_TOOL_ITERATIONS", 3)
        tool_trace = []

        for iteration in range(max_tool_iterations):
            request = {
                "model": self.model_name,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            }
            if _is_qwen_model(self.model_name):
                request.update({
                    "max_tokens": self.max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "presence_penalty": 1.5,
                    "extra_body": QWEN35_CHAT_EXTRA_BODY,
                })
            else:
                request["temperature"] = self.temperature
            response = _create_completion_with_context_retry(client, request)

            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                content = (message.content or "").strip()
                if content:
                    return {
                        "type": "final",
                        "answer": content.strip(),
                        "tool_trace": tool_trace,
                        "tool_info": tool_trace,
                    }
                break

            normalized_tool_calls = []
            decoded_tool_args = {}
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                raw_arguments = tool_call.function.arguments or "{}"
                try:
                    tool_args = json.loads(raw_arguments)
                    if not isinstance(tool_args, dict):
                        tool_args = {}
                except json.JSONDecodeError:
                    tool_args = {}

                if _is_qwen_model(self.model_name) and tool_name == "reasoning":
                    raw_candidate = tool_args.get("input")
                    candidate = (
                        str(raw_candidate).strip()
                        if raw_candidate is not None
                        else ""
                    )
                    if candidate and len(candidate) <= 4000:
                        canonical_input = candidate
                        tool_input = candidate
                    else:
                        canonical_input = "Solve the original user problem using the conversation context."
                        tool_input = prompt
                    canonical_arguments = json.dumps(
                        {"input": canonical_input},
                        ensure_ascii=False,
                    )
                    decoded_tool_args[tool_call.id] = {"input": tool_input}
                else:
                    canonical_arguments = raw_arguments
                    decoded_tool_args[tool_call.id] = tool_args

                normalized_tool_calls.append({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": canonical_arguments,
                    },
                })

            messages.append({
                "role": "assistant",
                "content": self._non_empty_text(message.content, "I will use a tool to solve the problem."),
                "tool_calls": normalized_tool_calls,
            })

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = decoded_tool_args.get(tool_call.id, {})
                tool_input = self._non_empty_text(
                    tool_args.get("input"),
                    prompt if tool_name == "reasoning" else "No tool input was provided.",
                )

                try:
                    if tool_name == "reasoning":
                        tool_result, tool_errors = self._reasoning_with_errors(tool_input)
                    elif tool_name == "coding":
                        tool_result = self.execute_code(tool_input)
                        tool_errors = []
                    else:
                        tool_result = "Unknown tool"
                        tool_errors = []
                except Exception as e:
                    tool_result = f"Error running {tool_name}: {type(e).__name__}: {str(e)[:200]}"
                    tool_errors = [tool_result]
                tool_message_content = self._non_empty_text(
                    tool_result,
                    f"{tool_name} failed after all retries.",
                )

                trace_entry = {
                    "iteration": iteration + 1,
                    "tool_call_id": tool_call.id,
                    "tool": tool_name,
                    "input": tool_input,
                    "result": tool_result,
                }
                if tool_errors:
                    trace_entry["error"] = self._format_errors(tool_errors)
                tool_trace.append(trace_entry)

                if tool_name == "reasoning" and not tool_result:
                    return self._empty_final_action(prompt, errors=tool_errors, tool_trace=tool_trace)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": tool_message_content,
                })

        messages.append({
            "role": "user",
            "content": "Use the prior reasoning tool results to produce the final answer now. Do not call any tools. Include only the final solution needed for verification."
        })
        request = {
            "model": self.model_name,
            "messages": messages,
        }
        if _is_qwen_model(self.model_name):
            request.update({
                "max_tokens": self.max_tokens,
                "temperature": 0.7,
                "top_p": 0.8,
                "presence_penalty": 1.5,
                "extra_body": QWEN35_CHAT_EXTRA_BODY,
            })
        else:
            request["temperature"] = self.temperature
        response = _create_completion_with_context_retry(client, request)
        content = (response.choices[0].message.content or "").strip()
        if content:
            return {
                "type": "final",
                "answer": content,
                "tool_trace": tool_trace,
                "tool_info": tool_trace,
            }

        return {
            "type": "final",
            "answer": "",
            "tool_trace": tool_trace,
            "tool_info": tool_trace,
        }
