import concurrent
import fcntl
from collections import defaultdict
from openai import OpenAI
from typing import List, Dict, Optional, Literal, Any
import json, os, re, time, warnings
import httpx
from lightmem.memory.prompts import EXTRACTION_PROMPTS, METADATA_GENERATE_PROMPT
from lightmem.configs.memory_manager.base_config import BaseMemoryManagerConfig
from lightmem.memory.utils import clean_response

model_name_context_windows = {
    "gpt-4o-mini": 128000,
    "qwen3-30b-a3b-instruct-2507": 128000,
    "glm-4.6": 200000,
    "DEFAULT": 128000,  # Recommended default context window
}

def _uses_responses_api(model: str) -> bool:
    return str(model or "").lower() == "gpt-5.4-mini"

def _is_retryable_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
        return True
    return type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }

def _acquire_responses_api_slot():
    lock_dir = os.getenv("LIGHTMEM_RESPONSES_LOCK_DIR")
    if not lock_dir:
        return None
    max_concurrency = max(1, int(os.getenv("LIGHTMEM_RESPONSES_MAX_CONCURRENCY", "1")))
    os.makedirs(lock_dir, exist_ok=True)
    while True:
        for slot_index in range(max_concurrency):
            lock_file = open(os.path.join(lock_dir, f"slot_{slot_index}.lock"), "a+")
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_file
            except BlockingIOError:
                lock_file.close()
        time.sleep(0.05)


def _release_responses_api_slot(lock_file):
    if lock_file is None:
        return
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()




def _responses_create_with_retry(client, params: Dict, max_attempts: int = 20):
    lock_file = _acquire_responses_api_slot()
    try:
        for attempt in range(1, max_attempts + 1):
            try:
                return client.responses.create(**params)
            except Exception as exc:
                if attempt >= max_attempts or not _is_retryable_api_error(exc):
                    raise
                delay = min(2 ** (attempt - 1), 30) + ((os.getpid() % 10) / 10.0)
                print(
                    "Retrying Responses API call after transient error: "
                    f"attempt={attempt}/{max_attempts}, delay={delay:.1f}s, "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError("Responses API retry loop exited unexpectedly")
    finally:
        _release_responses_api_slot(lock_file)




class OpenaiManager:
    def __init__(self, config: BaseMemoryManagerConfig):
        self.config = config

        if not self.config.model:
            self.config.model = "gpt-4o-mini"
        
        if self.config.model in model_name_context_windows:
            self.context_windows = model_name_context_windows[self.config.model]
        else:
            self.context_windows = model_name_context_windows["DEFAULT"]

        http_client = httpx.Client(verify=False)

        if os.environ.get("OPENROUTER_API_KEY"):  # Use OpenRouter
            self.client = OpenAI(
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                base_url=self.config.openrouter_base_url
                or os.getenv("OPENROUTER_API_BASE")
                or "https://openrouter.ai/api/v1",
            )
        else:
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = (
                self.config.openai_base_url
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_BASE_URL")
                or "https://api.openai.com/v1"
            )

            self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> Optional[str]:
        """
        Generate a response based on the given messages.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """
        use_responses_api = _uses_responses_api(self.config.model)
        if use_responses_api:
            params = {
                "model": self.config.model,
                "input": messages,
                "max_output_tokens": self.config.max_tokens,
            }
        else:
            params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
            }

        if not use_responses_api and os.getenv("OPENROUTER_API_KEY"):
            openrouter_params = {}
            
            models = getattr(self.config, 'models', None)    
            route = getattr(self.config, 'route', 'fallback') 
            if models:
                openrouter_params["models"] = models
                openrouter_params["route"] = route
                params.pop("model")

            if self.config.site_url and self.config.app_name:
                extra_headers = {
                    "HTTP-Referer": self.config.site_url,
                    "X-Title": self.config.app_name,
                }
                openrouter_params["extra_headers"] = extra_headers

            params.update(**openrouter_params)

        if response_format:
            if use_responses_api:
                params["text"] = {"format": response_format}
            else:
                params["response_format"] = response_format
        if not use_responses_api and getattr(self.config, "extra_body", None):
            params["extra_body"] = self.config.extra_body
        if tools:  # TODO: Remove tools if no issues found with new memory addition logic
            if use_responses_api:
                raise ValueError("Responses API tool calls are not used by the LightMem experiment pipeline")
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        max_attempts = (
            3
            if response_format and (use_responses_api or "deepseek" in str(self.config.model).lower())
            else 1
        )
        usage_info = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        parsed_response = None
        for attempt in range(max_attempts):
            if use_responses_api:
                response = _responses_create_with_retry(self.client, params)
                parsed_response = (getattr(response, "output_text", None) or "").strip()
            else:
                response = self.client.chat.completions.create(**params)
                parsed_response = self._parse_response(response, tools)
            response_usage = getattr(response, "usage", None)
            if use_responses_api:
                usage_info["prompt_tokens"] += int(getattr(response_usage, "input_tokens", 0) or 0)
                usage_info["completion_tokens"] += int(getattr(response_usage, "output_tokens", 0) or 0)
                usage_info["total_tokens"] += int(getattr(response_usage, "total_tokens", 0) or 0)
            else:
                for key in usage_info:
                    usage_info[key] += int(getattr(response_usage, key, 0) or 0)

            if not response_format:
                break

            candidate = parsed_response if isinstance(parsed_response, str) else ""
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", candidate.strip())
            cleaned_candidate = match.group(1).strip() if match else candidate.strip()
            try:
                json.loads(cleaned_candidate)
                break
            except (TypeError, json.JSONDecodeError) as exc:
                model_label = str(self.config.model)
                if attempt + 1 >= max_attempts:
                    print(
                        f"{model_label} JSON response remained invalid after "
                        f"{max_attempts} attempts: {exc}"
                    )
                    break
                budget_key = "max_output_tokens" if use_responses_api else "max_tokens"
                previous_budget = int(params.get(budget_key) or 4096)
                next_budget = min(previous_budget * 2, 16384)
                if use_responses_api:
                    finish_reason = getattr(response, "status", None) or getattr(response, "incomplete_details", None)
                else:
                    finish_reason = getattr(response.choices[0], "finish_reason", None)
                print(
                    f"Retrying invalid {model_label} JSON response: "
                    f"attempt={attempt + 1}/{max_attempts}, "
                    f"finish_reason={finish_reason}, "
                    f"max_tokens={previous_budget}->{next_budget}"
                )
                params[budget_key] = next_budget

        return parsed_response, usage_info

    def meta_text_extract(
        self,
        extract_list: List[List[List[Dict]]],
        messages_use: Literal["user_only", "assistant_only", "hybrid"] = "user_only",
        topic_id_mapping: Optional[List[List[int]]] = None,
        extraction_mode: Literal["flat", "event"] = "flat",
        custom_prompts: Optional[Dict[str, str]] = None  
    ) -> List[Optional[Dict]]:
        """
        Extract metadata from text segments using parallel processing.

        Args:
            extract_list: List of message segments to process
            messages_use: Strategy for which messages to use
            topic_id_mapping: For each API call, the global topic IDs
            extraction_mode: "flat" or "event"
            custom_prompts: Optional custom prompts. If None, use defaults from EXTRACTION_PROMPTS

        Returns:
            List of extracted metadata results, None for failed segments
        """
        if not extract_list:
            return []
        
        default_prompts = EXTRACTION_PROMPTS.get(extraction_mode, {})
        
        if custom_prompts is None:
            prompts = default_prompts
        else:
            prompts = {**default_prompts, **custom_prompts}
        
        if extraction_mode == "flat":
            return self._extract_with_prompt(
                system_prompt=prompts.get("factual", METADATA_GENERATE_PROMPT),
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="factual"
            )
        
        elif extraction_mode == "event":
            factual_results = self._extract_with_prompt(
                system_prompt=prompts["factual"],
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="factual"
            )
            
            relational_results = self._extract_with_prompt(
                system_prompt=prompts["relational"],
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="relational"
            )
            
            return self._merge_dual_perspective_results(
                factual_results, 
                relational_results
            )
        
        else:
            raise ValueError(f"Unknown extraction_mode: {extraction_mode}")
    
    def _merge_dual_perspective_results(
        self,
        factual_results: List[Optional[Dict]],
        relational_results: List[Optional[Dict]]
    ) -> List[Optional[Dict]]:
        """
        Args:
            factual_results: Factual extraction results
            relational_results: Relational extraction results
        
        Returns:
            Merged results with combined cleaned_result and accumulated usage
        """
        merged_results = []
        
        for factual, relational in zip(factual_results, relational_results):
            if factual is None and relational is None:
                merged_results.append(None)
                continue
            
            merged = {
                "input_prompt": [],
                "output_prompt": "",
                "cleaned_result": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
            if factual is not None:
                merged["input_prompt"].extend(factual.get("input_prompt", []))
                merged["cleaned_result"].extend(factual.get("cleaned_result", []))
                if factual.get("usage"):
                    for key in merged["usage"]:
                        merged["usage"][key] += factual["usage"].get(key, 0)
            
            if relational is not None:
                merged["input_prompt"].extend(relational.get("input_prompt", []))
                merged["cleaned_result"].extend(relational.get("cleaned_result", []))
                if relational.get("usage"):
                    for key in merged["usage"]:
                        merged["usage"][key] += relational["usage"].get(key, 0)
            
            merged["output_prompt"] = (
                f"Factual: {factual.get('output_prompt', 'N/A') if factual else 'N/A'}\n"
                f"Relational: {relational.get('output_prompt', 'N/A') if relational else 'N/A'}"
            )
            
            merged_results.append(merged)
        
        return merged_results

    def _extract_with_prompt(
        self,
        system_prompt: str,
        extract_list: List[List[List[Dict]]],
        messages_use: str,
        topic_id_mapping: Optional[List[List[int]]],
        entry_type: str = "factual"
    ) -> List[Optional[Dict]]:
        """
        Args:
            system_prompt: System prompt for extraction
            extract_list: List of message segments
            messages_use: Message filtering strategy
            topic_id_mapping: Global topic IDs
            entry_type: "factual" or "relational"
        
        Returns:
            List of extraction results
        """
        def concatenate_messages(segment: List[Dict], messages_use: str) -> str:
            """Concatenate messages based on usage strategy"""
            role_filter = {
                "user_only": {"user"},
                "assistant_only": {"assistant"},
                "hybrid": {"user", "assistant"}
            }

            if messages_use not in role_filter:
                raise ValueError(f"Invalid messages_use value: {messages_use}")

            allowed_roles = role_filter[messages_use]
            message_lines = []

            for mes in segment:
                if mes.get("role") in allowed_roles:
                    sequence_id = mes["sequence_number"]
                    role = mes["role"]
                    content = mes.get("content", "")
                    speaker_name = mes.get("speaker_name", "")
                    time_stamp = mes.get("time_stamp", "")
                    weekday = mes.get("weekday", "")
                    
                    time_prefix = ""
                    if time_stamp and weekday:
                        time_prefix = f"[{time_stamp}, {weekday}] "

                    if speaker_name:
                        message_lines.append(f"{time_prefix}{sequence_id//2}.{speaker_name}: {content}")
                    else:
                        message_lines.append(f"{time_prefix}{sequence_id//2}.{role}: {content}")
            
            return "\n".join(message_lines)

        max_workers = min(len(extract_list), 5)

        def process_segment_wrapper(args):
            api_call_idx, api_call_segments = args
            try:
                user_prompt_parts: List[str] = []
                
                global_topic_ids: List[int] = []
                if topic_id_mapping and api_call_idx < len(topic_id_mapping):
                    global_topic_ids = topic_id_mapping[api_call_idx]

                for topic_idx, topic_segment in enumerate(api_call_segments):
                    if topic_idx < len(global_topic_ids):
                        global_topic_id = global_topic_ids[topic_idx]
                    else:
                        global_topic_id = topic_idx + 1
                    
                    topic_text = concatenate_messages(topic_segment, messages_use)
                    user_prompt_parts.append(f"--- Topic {global_topic_id} ---\n{topic_text}")

                print(f"User prompt for API call {api_call_idx}:\n" + "\n".join(user_prompt_parts))
                user_prompt = "\n".join(user_prompt_parts)
                
                metadata_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                
                raw_response, usage_info = self.generate_response(
                    messages=metadata_messages,
                    response_format={"type": "json_object"},
                )
                metadata_facts = clean_response(raw_response)
                
                for entry in metadata_facts:
                    entry["entry_type"] = entry_type

                return {
                    "input_prompt": metadata_messages,
                    "output_prompt": raw_response,
                    "cleaned_result": metadata_facts,
                    "usage": usage_info,
                    "entry_type": entry_type
                }
                
            except Exception as e:
                print(f"Error processing API call {api_call_idx}: {e}", flush=True)
                if _is_retryable_api_error(e):
                    raise
                return {
                    "input_prompt": [],
                    "output_prompt": "",
                    "cleaned_result": [],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "entry_type": entry_type
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                results = list(executor.map(process_segment_wrapper, enumerate(extract_list)))
            except Exception as e:
                print(f"Error in parallel processing: {e}", flush=True)
                raise

        return results

    def _call_update_llm(self, system_prompt, target_entry, candidate_sources):
        target_memory = target_entry["payload"]["memory"]
        candidate_memories = [c["payload"]["memory"] for c in candidate_sources]

        user_prompt = (
            f"Target memory:{target_memory}\n"
            f"Candidate memories:\n" + "\n".join([f"- {m}" for m in candidate_memories])
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response_text, usage_info = self.generate_response(
            messages=messages,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response_text)
            if "action" not in result:
                result = {"action": "ignore"}
            result["usage"] = usage_info  
            return result
        except Exception:
            return {"action": "ignore", "usage": usage_info if 'usage_info' in locals() else None}
