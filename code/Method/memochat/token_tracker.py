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
    一个用于在复杂Python项目中跟踪LLM token消耗和执行时间的工具。

    它使用上下文管理器来定义嵌套的统计阶段，并通过猴子补丁来
    自动拦截和计算LLM API调用的token消耗。同时记录每个阶段的执行时间。

    使用方法:
    1. 实例化: tracker = TokenTracker()
    2. 补丁API: tracker.patch_llm_api("openai.ChatCompletion.create")
    3. 使用阶段:
       with tracker.stage("数据处理"):
           ...
           with tracker.stage("数据清洗"):
               ...
    4. 程序结束时，结果会自动保存到 'token_usage.json'。
    """
    def __init__(self, output_file: str = 'token_usage.json'):
        """
        初始化追踪器。

        :param output_file: 结果输出的JSON文件名。
        """
        self.output_file = output_file
        # 使用线程局部存储来确保多线程环境下的线程安全
        self._context_stack = threading.local()
        # 初始化根节点
        self.root = self._create_stage_node('root')
        self._context_stack.value = [self.root]
        
        # 注册程序退出时的保存函数
        atexit.register(self.save_to_json)
        print("TokenTracker initialized. Output will be saved to", self.output_file)

    def _create_stage_node(self, name: str) -> Dict[str, Any]:
        """创建一个新的阶段节点"""
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
        通过猴子补丁替换原始LLM API调用函数。

        :param function_path: LLM函数的完整路径字符串, e.g., "openai.resources.chat.completions.Completions.create"
        :param token_extractor: 一个可选函数，用于从API响应中提取token总数。
                                 默认为OpenAI的格式。
        """
        if token_extractor is None:
            # 默认的OpenAI token提取器
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
            raise ImportError(f"无法找到或导入函数: {function_path}. 请确保路径正确且库已安装。") from e

        @functools.wraps(original_llm_call)
        def _token_counting_wrapper(*args, **kwargs):
            # 调用原始函数
            response = original_llm_call(*args, **kwargs)
            
            # 提取token
            try:
                tokens_dict = token_extractor(response)
            except Exception as e:
                print(f"Warning: Token extractor failed for response: {response}. Error: {e}")
                tokens_dict = {}

            # 更新当前阶段的token计数
            if self._context_stack.value:
                current_stage = self._context_stack.value[-1]
                current_stage['prompt_tokens'] += tokens_dict.get('prompt_tokens', 0)
                current_stage['completion_tokens'] += tokens_dict.get('completion_tokens', 0)
                current_stage['total_tokens'] += tokens_dict.get('total_tokens', 0)
            else:
                # 如果栈为空（理论上不应该发生，因为有root），则记录到root
                self.root['prompt_tokens'] += tokens_dict.get('prompt_tokens', 0)
                self.root['completion_tokens'] += tokens_dict.get('completion_tokens', 0)
                self.root['total_tokens'] += tokens_dict.get('total_tokens', 0)

            return response

        # 应用猴子补丁
        setattr(target_object, function_name, _token_counting_wrapper)
        print(f"Successfully patched '{function_path}'. Token tracking is active.")

    @contextlib.contextmanager
    def stage(self, name: str):
        """
        一个上下文管理器，用于定义一个统计阶段。
        
        :param name: 阶段的名称。
        """
        parent_stage = self._context_stack.value[-1]
        
        if name in parent_stage['sub_stages']:
            # 如果同名阶段已存在，直接复用
            new_stage = parent_stage['sub_stages'][name]
        else:
            # 否则创建新阶段
            new_stage = self._create_stage_node(name)
            parent_stage['sub_stages'][name] = new_stage

        # 记录开始时间
        start_time = time.time()
        new_stage['start_time'] = start_time

        self._context_stack.value.append(new_stage)
        try:
            yield
        finally:
            # 记录结束时间并计算持续时间
            end_time = time.time()
            new_stage['end_time'] = end_time
            new_stage['duration_seconds'] += end_time - start_time
            self._context_stack.value.pop()

    def save_to_json(self):
        """将统计结果保存到JSON文件。"""
        
        def aggregate_tokens(node: Dict[str, Any]):
            """
            使用后序遍历，递归地将子节点的token计数和时间聚合到父节点。
            """
            # 1. 对所有子节点进行递归聚合
            for sub_node in node['sub_stages'].values():
                aggregate_tokens(sub_node)
            
            # 2. 将(已经聚合了其后代数据的)子节点的token总数加到当前节点
            node['prompt_tokens'] += sum(s['prompt_tokens'] for s in node['sub_stages'].values())
            node['completion_tokens'] += sum(s['completion_tokens'] for s in node['sub_stages'].values())
            node['total_tokens'] += sum(s['total_tokens'] for s in node['sub_stages'].values())

        # 从根节点开始，递归地聚合所有token计数
        aggregate_tokens(self.root)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.root, f, indent=4, ensure_ascii=False)
        print(f"\nToken usage statistics saved to '{self.output_file}'")