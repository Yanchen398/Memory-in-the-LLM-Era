import json
import atexit
import contextlib
import functools
import importlib
import threading
import time
from typing import Callable, Any, Dict, Optional

class TokenTracker:
    """
    A utility for tracking LLM token usage and execution time in complex Python projects.

    It uses context managers to define nested tracking stages and monkey-patches
    LLM API calls to automatically intercept and count token usage. It also records
    execution time for each stage.

    Usage:
    1. Instantiate: tracker = TokenTracker()
    2. Patch the API: tracker.patch_llm_api("openai.ChatCompletion.create")
    3. Use stages:
       with tracker.stage("data processing"):
           ...
           with tracker.stage("data cleaning"):
               ...
    4. At program exit, results are automatically saved to 'token_usage.json'.
    """
    def __init__(self, output_file: str = 'token_usage.json'):
        """
        Initialize the tracker.

        :param output_file: Name of the JSON output file.
        """
        self.output_file = output_file
        # Use thread-local storage to ensure thread safety in multi-threaded environments.
        self._context_stack = threading.local()
        # Initialize the root node.
        self.root = self._create_stage_node('root')
        self._context_stack.value = [self.root]
        
        # Register the save function to run at process exit.
        atexit.register(self.save_to_json)
        print("TokenTracker initialized. Output will be saved to", self.output_file)

    def _create_stage_node(self, name: str) -> Dict[str, Any]:
        """Create a new stage node."""
        return {
            "name": name,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0.0,
            "sub_stages": {}
        }

    def patch_llm_api(self, 
                      function_path: str = "openai.resources.chat.completions.Completions.create", 
                      token_extractor: Optional[Callable[[Any], Dict[str, int]]] = None):
        """
        Replace the original LLM API call with a monkey-patched wrapper.

        :param function_path: Full dotted path to the LLM function,
                              e.g. "openai.resources.chat.completions.Completions.create"
        :param token_extractor: Optional function used to extract token counts from the API response.
                                Defaults to the OpenAI response format.
        """
        if token_extractor is None:
            # Default token extractor for OpenAI responses.
            def default_extractor(response):
                usage = response.usage
                return {
                    'prompt_tokens': usage.prompt_tokens,
                    'completion_tokens': usage.completion_tokens,
                    'total_tokens': usage.total_tokens
                }
            token_extractor = default_extractor

        try:
            class_path, function_name = function_path.rsplit('.', 1)
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            target_object = getattr(module, class_name)
            original_llm_call = getattr(target_object, function_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not find or import function: {function_path}. Please verify the path and install the required library.") from e

        @functools.wraps(original_llm_call)
        def _token_counting_wrapper(*args, **kwargs):
            # Call the original function.
            response = original_llm_call(*args, **kwargs)
            
            # Extract token counts.
            try:
                tokens_dict = token_extractor(response)
            except Exception as e:
                print(f"Warning: Token extractor failed for response: {response}. Error: {e}")
                tokens_dict = {}

            # Update token counts for the current stage.
            if self._context_stack.value:
                current_stage = self._context_stack.value[-1]
                current_stage['prompt_tokens'] += tokens_dict.get('prompt_tokens', 0)
                current_stage['completion_tokens'] += tokens_dict.get('completion_tokens', 0)
                current_stage['total_tokens'] += tokens_dict.get('total_tokens', 0)
            else:
                # If the stack is empty, record token usage on the root node.
                self.root['prompt_tokens'] += tokens_dict.get('prompt_tokens', 0)
                self.root['completion_tokens'] += tokens_dict.get('completion_tokens', 0)
                self.root['total_tokens'] += tokens_dict.get('total_tokens', 0)

            return response

        # Apply the monkey patch.
        setattr(target_object, function_name, _token_counting_wrapper)
        print(f"Successfully patched '{function_path}'. Token tracking is active.")

    @contextlib.contextmanager
    def stage(self, name: str):
        """
        A context manager used to define a tracking stage.
        
        :param name: Name of the stage.
        """
        parent_stage = self._context_stack.value[-1]
        
        if name in parent_stage['sub_stages']:
            # Reuse the stage if one with the same name already exists.
            new_stage = parent_stage['sub_stages'][name]
        else:
            # Otherwise create a new stage.
            new_stage = self._create_stage_node(name)
            parent_stage['sub_stages'][name] = new_stage

        # Record the start time.
        start_time = time.time()
        new_stage['start_time'] = start_time

        self._context_stack.value.append(new_stage)
        try:
            yield
        finally:
            # Record the end time and compute the duration.
            end_time = time.time()
            new_stage['end_time'] = end_time
            new_stage['duration_seconds'] += end_time - start_time
            self._context_stack.value.pop()

    def save_to_json(self):
        """Save tracking results to a JSON file."""
        
        def aggregate_tokens(node: Dict[str, Any]):
            """
            Recursively aggregate token counts from child nodes into the parent node.
            """
            # 1. Recursively aggregate all child nodes.
            for sub_node in node['sub_stages'].values():
                aggregate_tokens(sub_node)
            
            # 2. Add the aggregated child token totals to the current node.
            node['prompt_tokens'] += sum(s['prompt_tokens'] for s in node['sub_stages'].values())
            node['completion_tokens'] += sum(s['completion_tokens'] for s in node['sub_stages'].values())
            node['total_tokens'] += sum(s['total_tokens'] for s in node['sub_stages'].values())

        # Start from the root node and aggregate all token counts recursively.
        aggregate_tokens(self.root)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.root, f, indent=4, ensure_ascii=False)
        print(f"\nToken usage statistics saved to '{self.output_file}'")
