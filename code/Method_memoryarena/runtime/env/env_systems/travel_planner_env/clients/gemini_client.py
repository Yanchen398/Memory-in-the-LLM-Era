"""
Gemini Client - Implementation of BaseModelClient for Google Gemini API
Supports Gemini's function calling API
"""

import os
import json
import uuid
from typing import List, Dict, Optional

from .base_client import BaseModelClient, ModelResponse, ToolCall

import google.generativeai as genai
from google.generativeai.types import generation_types



class GeminiClient(BaseModelClient):
    """
    Google Gemini client with function calling support.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: str = None):
        super().__init__(model_name)
        
        # Configure API key
        api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name)
        self._current_chat = None
    
    def _convert_tools_to_gemini_format(self, tools: List[Dict]) -> List:
        if not tools:
            return None
        
        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                function_declarations.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {})
                })
        
        if not function_declarations:
            return None
        
        # All functions go into ONE function_declarations array
        return [{"function_declarations": function_declarations}]
    
    def _convert_messages_to_gemini_format(self, messages: List[Dict]) -> tuple:
        system_instruction = None
        history = []
        current_content = None
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if role == "system":
                system_instruction = content
            elif role == "user":
                current_content = content
                # Also add to history for context
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                if content:
                    history.append({"role": "model", "parts": [content]})
            elif role == "tool":
                # Tool results - add as user message with function response
                # Note: "name" field must be provided in format_tool_result for Gemini
                tool_name = msg.get("name", "unknown_function")
                history.append({
                    "role": "user",
                    "parts": [{
                        "function_response": {
                            "name": tool_name,
                            "response": {"result": content}
                        }
                    }]
                })
        
        if history and history[-1].get("role") == "user":
            last_parts = history[-1].get("parts", [])
            is_tool_result = any(
                isinstance(p, dict) and "function_response" in p 
                for p in last_parts
            )
            if not is_tool_result:
                history = history[:-1]
        
        return system_instruction, history, current_content
    
    def chat_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 8192
    ) -> ModelResponse:
        try:
            # Convert tools to Gemini format
            gemini_tools = self._convert_tools_to_gemini_format(tools)
            
            # Convert messages
            system_instruction, history, current_content = self._convert_messages_to_gemini_format(messages)
            
            # Create model with system instruction if provided
            if system_instruction:
                model = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_instruction
                )
            else:
                model = self.model
            
            # Configure generation
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            # Start or continue chat
            chat = model.start_chat(history=history)
            
            # Send message with tools
            if gemini_tools:
                # Use tool_config to control function calling behavior
                tool_config = {"function_calling_config": {"mode": "AUTO"}}
                response = chat.send_message(
                    current_content or "Continue",
                    generation_config=generation_config,
                    tools=gemini_tools,
                    tool_config=tool_config
                )
            else:
                response = chat.send_message(
                    current_content or "Continue",
                    generation_config=generation_config
                )
            
            # Track usage (Gemini provides usage metadata)
            if hasattr(response, 'usage_metadata'):
                self.total_input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
                self.total_output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            # Parse response
            content = None
            tool_calls = None
            
            # Check for function calls
            if response.candidates and response.candidates[0].content.parts:
                parts = response.candidates[0].content.parts
                text_parts = []
                func_calls = []
                
                for part in parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        func_calls.append(ToolCall(
                            id=str(uuid.uuid4()),  # Gemini doesn't provide IDs
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {}
                        ))
                
                if text_parts:
                    content = "\n".join(text_parts)
                if func_calls:
                    tool_calls = func_calls
            
            # Convert raw response to dict for serialization
            # This captures ALL parts of the Gemini response for trace purposes
            try:
                raw_response_dict = response.to_dict()
            except Exception:
                # Fallback: manually extract what we can
                raw_response_dict = {
                    "text": response.text if hasattr(response, 'text') else None,
                    "candidates": str(response.candidates) if hasattr(response, 'candidates') else None
                }
            
            return ModelResponse(
                content=content,
                tool_calls=tool_calls,
                raw_response=raw_response_dict
            )
            
        except generation_types.StopCandidateException as e:
            # Handle MALFORMED_FUNCTION_CALL and similar errors
            print(f"Gemini API error: {e}")
            # Return empty response instead of crashing
            return ModelResponse(
                content=None,
                tool_calls=None,
                raw_response={"error": str(e), "finish_reason": "MALFORMED_FUNCTION_CALL"}
            )
        except Exception as e:
            print(f"Gemini API error: {e}")
            raise
    
    def format_tool_result(self, tool_call_id: str, result: str, name: str = None) -> Dict:
        """
        Format tool result as a message for Gemini.
        
        Note: Gemini requires the function name in the response.
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name or "unknown_function",  # Gemini needs this
            "content": result
        }
    
    def format_assistant_tool_calls(self, tool_calls: List[ToolCall]) -> Dict:
        """
        Format assistant's tool calls as a message.
        """
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in tool_calls
            ]
        }
    
    def get_usage_stats(self) -> Dict:
        """Get usage stats"""
        # Gemini pricing (approximate, may vary)
        # gemini-1.5-flash: $0.075/1M input, $0.30/1M output
        # gemini-2.0-flash: similar pricing
        input_cost_per_1m = 0.075
        output_cost_per_1m = 0.30
        
        total_cost = (
            self.total_input_tokens * input_cost_per_1m / 1_000_000 +
            self.total_output_tokens * output_cost_per_1m / 1_000_000
        )
        
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_cost': total_cost
        }


# For testing
if __name__ == "__main__":
    client = GeminiClient(model_name="gemini-2.0-flash")
    
    messages = [
        {"role": "system", "content": "You are a travel planning assistant."},
        {"role": "user", "content": "Search for flights from New York to Los Angeles on 2024-03-15"}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "FlightSearch",
                "description": "Search for flights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                        "date": {"type": "string"}
                    },
                    "required": ["origin", "destination", "date"]
                }
            }
        }
    ]
    
    response = client.chat_with_tools(messages, tools)
    print(f"Content: {response.content}")
    print(f"Tool calls: {response.tool_calls}")
    print(f"Usage: {client.get_usage_stats()}")
