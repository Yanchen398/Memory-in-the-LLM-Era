from typing import Any, Dict, Optional, Tuple
import json
from .base_env import BaseEnvironment
from .formal_reasoning_env.llm_backend import create_backend, LLMBackend
import os
import pdb

class MathEnvironment(BaseEnvironment):
    """Math environment for mathematical problem-solving tasks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.history = []
        self._current_observation = None
        self.step_count = 0
        self.model_name = self.config.get("model_name", "gpt-5-mini")
        self.temperature = self.config.get("temperature", 0.0)
        self.max_tokens = self.config.get("max_tokens", 2000)
        
        # Initialize LLM backend
        backend_name = self.config.get("backend", "openai")
        self.llm_backend = create_backend(
            backend_name=backend_name,
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=self.config.get("base_url", None)
        )
        self.tools = {
            "reasoning": {
                "name": "reasoning",
                "description": "Solve a math problem and return the final answer.",
                "input_schema": {"type": "object", "properties": {"task": {"type": "string"}}},
            },
            "coding": {
                "name": "coding",
                "description": "Execute Python code to solve computational problems. Returns the code output or error message.",
                "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}},
            }
        }
        print("MathEnvironment initialized with config")

        
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        self.history = []
        self.step_count = 0
        self._current_observation = {
            "task": None,
            "step": self.step_count,
            "state": {"history_len": 0},
            "tool_calls": [],
            "tool_results": [],
            "final": None,
        }
        return self._current_observation
    def judge(
        self,
        action: str,
        ground_truth: Optional[str] = None,
        query: Optional[str] = None,
    ) -> bool:
        """
        Judge if the model's answer is mathematically equivalent to the ground truth.
        
        Args:
            action: The model's answer
            ground_truth: The correct answer
            query: The original question (optional, for context)
        
        Returns:
            bool: True if the answers are mathematically equivalent, False otherwise
        """
        judge_prompt = f"""
            You are a math expert. 
            Determine if these two expressions are mathematically equivalent answer for the given question:
            Question: {query}
            Expression 1: {action}
            Expression 2: {ground_truth}

            Respond only with "yes" or "no". """
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that judges the equivalence of two mathematical expressions."},
            {"role": "user", "content": judge_prompt}
        ]
        output = self.llm_backend.chat(
            messages=messages,
            temperature=self.temperature
        ).lower()
        return "yes" in output, output
    
    def reasoning(self, task: str) -> str:
        prompt = f"Solve the following math problem and provide the final answer only.\n\nProblem: {task}\n"
        messages = [
            {"role": "system", "content": "You are a careful math solver."},
            {"role": "user", "content": prompt},
        ]
        return self.llm_backend.chat(messages=messages)

    def _parse_action(self, action: Any) -> Dict[str, Any]:
        """Parse an action and its tool execution results."""
        if isinstance(action, dict):
            return action
        if isinstance(action, str):
            try:
                parsed = json.loads(action)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {"type": "final", "answer": action}

    def step(
        self,
        action: Any,
        ground_truth: Any = None,
        need_judge: bool = False,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Process action result from agent.
        The agent has already executed tools, so we just process and store the result.
        """
        parsed = self._parse_action(action)
        self.step_count += 1

        observation: Dict[str, Any]
        reward = None
        tool_info = {}
        tool_trace = []
        final_answer = None
        # Handle tool execution results from agent
        if parsed.get("type") == "tool_result":
            tool_name = parsed.get("tool")
            tool_input = parsed.get("input")
            tool_result = parsed.get("result")
            
            tool_info = {
                "tool": tool_name,
                "input": tool_input,
                "result": tool_result
            }
            tool_trace = [tool_info]
            final_answer = tool_result
            
        # Handle final answers
        elif parsed.get("type") == "final":
            final_answer = parsed.get("answer", parsed)
            tool_trace = parsed.get("tool_trace") or parsed.get("tool_info") or []
            tool_info = parsed.get("tool_info") or tool_trace or {}
        else:
            final_answer = parsed.get("answer", parsed)
            tool_trace = parsed.get("tool_trace") or parsed.get("tool_info") or []
            tool_info = parsed.get("tool_info") or tool_trace or {}

        

        if need_judge and ground_truth is not None:
            reward, judge_result = self.judge(str(final_answer), str(ground_truth))
        action_input = parsed.get("input", "") if isinstance(parsed, dict) else ""
        if "<memory_context>" in action_input:
            memory_context = action_input.split("<memory_context>")[1].split("</memory_context>")[0]
        else:
            memory_context = None
        observation = {
            "step": self.step_count,
            "state": {"history_len": len(self.history)},
            "tool_info": tool_info,
            "tool_trace": tool_trace,
            "final": final_answer,
            "memory_context": memory_context,
            "reward": reward,
            "judge_result": judge_result if need_judge and ground_truth is not None else None,
          
        }
        
        self.history.append(
            {
                "step": self.step_count,
                "action_type": parsed.get("type"),
                "tool": parsed.get("tool"),
                "tool_result": tool_info,
                "tool_trace": tool_trace,
                "memory_context": memory_context,
                "final": final_answer,
                "reward": reward,
                "judge_result": judge_result if need_judge and ground_truth is not None else None,
            }
        )
        self._current_observation = observation
        return observation, reward, {}
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        return self._current_observation or self.reset()
    def close(self):
        """Cleanup math environment."""
        self.history = []
        self._current_observation = None
