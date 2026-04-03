import argparse
import os
import sys
import time
from .config import globalconfig, load_config
from .utils import retrieve, generation
from .dataloader import Dataloder
from .structure import build_tree, save_tree, load_tree, MemTree
from .token_tracker import TokenTracker
import concurrent.futures
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

DEFAULT_TOKEN_FILE = "/home/docker/IndepthMem/Result/LONGMEMEVAL/memtree/token_tracker.json"


def ensure_parent_dir(path):
    if path:
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

def create_sample_config(base_config, sample_index):
    """
    为指定的样本创建独立的配置
    
    Args:
        base_config: 基础配置对象
        sample_index: 样本索引
    
    Returns:
        新的配置对象，包含独立的数据库设置
    """
    from types import SimpleNamespace
    from pymilvus import MilvusClient
    from .config import clean_str, create_collections, get_embedding_model
    
    # 复制基础配置的所有属性
    config_dict = {}
    for attr in dir(base_config):
        if not attr.startswith('_'):
            config_dict[attr] = getattr(base_config, attr)
    
    # 创建新的配置对象
    sample_config = SimpleNamespace(**config_dict)
    
    # 创建独立的数据库名称
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data", base_config.dataset_name)
    sample_db_name = os.path.join(data_dir, f'{clean_str(base_config.embedding_model_name).replace(" ", "")}_sample_{sample_index}_{base_config.vdb_name}')
    
    # 更新配置
    sample_config.db_name = sample_db_name
    sample_config.collection_name = f"{base_config.collection_name}_sample_{sample_index}"
    # sample_config.save_path = os.path.join(data_dir, f"{base_config.save_name}_sample_{sample_index}")
    sample_config.save_path = os.path.join(data_dir, f"{base_config.save_name}")
    
    # 创建新的 Milvus 客户端和集合
    sample_config.client = MilvusClient(sample_config.db_name)
    create_collections(sample_config.client, sample_config.collection_name, base_config.dimension)
    
    # 保持相同的嵌入模型（不需要重新加载）
    sample_config.model = base_config.model
    
    print(f"Created independent database for sample {sample_index}: {sample_config.db_name}")
    
    return sample_config

def update_global_config(new_config):
    """
    临时更新全局配置以供其他模块使用
    
    Args:
        new_config: 新的配置对象
    """
    # 导入并更新全局配置
    from . import config
    
    if config.globalconfig is None:
        print("Warning: globalconfig is None, cannot update")
        return
    
    # 更新全局配置对象的属性 - 只更新实际的配置属性
    core_attrs = ['db_name', 'collection_name', 'client', 'model', 'save_path',
                  'dataset_name', 'dimension', 'embedding_model_name', 'vdb_name',
                  'embedding_batch_size', 'llm_parallel_nums', 'base_threshold',
                  'rate', 'max_depth', 'top_k_retrieve']
    
    updated_attrs = []
    for attr in core_attrs:
        if hasattr(new_config, attr):
            setattr(config.globalconfig, attr, getattr(new_config, attr))
            updated_attrs.append(attr)
    
    print(f"Updated global config attributes: {updated_attrs}")
    print(f"Updated global config with new database: {new_config.db_name}")
    
    # 验证更新是否成功
    if hasattr(config.globalconfig, 'collection_name'):
        print(f"Verification: globalconfig.collection_name = {config.globalconfig.collection_name}")
    else:
        print("Warning: collection_name still missing after update")

def split_into_batches(total_samples, batch_size):
    """将样本索引分割成批次"""
    batches = []
    for i in range(0, total_samples, batch_size):
        batch = list(range(i, min(i + batch_size, total_samples)))
        batches.append(batch)
    return batches

def process_batch(batch_indices, global_config, token_file, batch_id):
    """
    处理单个批次的样本
    
    Args:
        batch_indices: 当前批次包含的样本索引列表
        global_config: 全局配置
        token_file: token追踪文件路径
        batch_id: 批次ID
    
    Returns:
        当前批次的处理结果
    """
    start_time = time.time()
    from .dataloader import Dataloder
    from .token_tracker import TokenTracker
    
    # 为当前进程创建独立的token追踪文件
    batch_token_file = f"{token_file.replace('.json', '')}_batch_{batch_id}.json"
    ensure_parent_dir(batch_token_file)
    tracker = TokenTracker(output_file=batch_token_file)
    tracker.patch_llm_api()
    
    print(f"Batch {batch_id}: Processing samples {batch_indices}")
    
    # 创建数据加载器
    dataloader = Dataloder(global_config)
    batch_results = []
    
    for i in batch_indices:
        print(f"Batch {batch_id}: Processing sample {i}")
        
        # 创建当前样本的独立配置
        sample_config = create_sample_config(global_config, i)
        
        # 更新数据加载器的配置
        dataloader.update_config(sample_config)
        
        # 临时更新全局配置
        update_global_config(sample_config)
        
        # 构建树结构
        print(f"Batch {batch_id}: building tree for index {i}")
        with tracker.stage(f"Sample {i}"):
            tree = load_tree(sample_config.save_path, i)
            questions, sessions = dataloader.data[i]
            
            if tree is None:
                tree = MemTree("")
                root_id = id(tree.root)
                for session_id, session in sessions.items():
                    with tracker.stage(f"Session {session_id}"):
                        dial_id = 0
                        for dial in session:
                            with tracker.stage(f"Dialog {dial_id}"):
                                tree.add_node(dial, root_id)
                            dial_id += 1
                save_tree(tree, sample_config.save_path, i)
        
        # 获取问题数据
        questions = dataloader.data[i][0]
        
        # 检索相关内容
        print(f"Batch {batch_id}: retrieving data for sample {i}")
        retrieve_result = retrieve(questions, i, batch_token_file)
        
        # 生成回答
        result = generation(tree, retrieve_result, i, batch_token_file)
        
        # 格式化结果
        qa_list = []
        for question, context, answer in result:
            if isinstance(question, dict):
                question_text = question.get("question", "")
                expected_answer = question.get("answer", "")
                category = question.get("category", "")
            else:
                question_text = str(question)
                expected_answer = ""
                category = ""
            
            retrieved = context.split('\n\n') if context else []
            
            qa_item = {
                "question": question_text,
                "answer": expected_answer,
                "category": category,
                "response": answer,
                "retrieved": retrieved
            }
            qa_list.append(qa_item)
        
        batch_result = {
            "sample_id": dataloader.sample_ids[i],
            "qa": qa_list
        }
        batch_results.append(batch_result)
        
        print(f"Batch {batch_id}: Completed sample {i}")
    
    end_time = time.time()
    print(f"Batch {batch_id}: Completed all samples {batch_indices}，时间消耗: {end_time - start_time:.2f}秒")
    return batch_results

def run_memtree(config_path=None, dataset_name=None, dataset_path=None, output_path=None,
                token_file=None,
                batch_size=None, num_processes=None):
    """
    运行 MemoryTree 的主要函数
    
    Args:
        config_path: 配置文件路径
        dataset_name: 数据集名称
        output_path: 输出文件路径
        batch_size: 每个批次的样本数量，如果为None则不使用多进程
        num_processes: 进程数量，如果为None则使用CPU核心数
    
    Returns:
        处理结果
    """
    
    if config_path:
        from types import SimpleNamespace
        import yaml
        from .config import GlobalConfig, resolve_memtree_config_dict
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if dataset_name:
            config['dataset_name'] = dataset_name
        if dataset_path:
            config['dataset_path'] = dataset_path
        if output_path:
            config['output_path'] = output_path
        if token_file:
            config['token_file'] = token_file
        if batch_size is not None:
            config['batch_size'] = batch_size
        if num_processes is not None:
            config['num_processes'] = num_processes

        config['config_path'] = config_path
        config = resolve_memtree_config_dict(config, config_path)
        output_path = config.get('output_path')
        token_file = config.get('token_file', token_file)
        batch_size = config.get('batch_size', 1)
        num_processes = config.get('num_processes', num_processes)
            
        config = SimpleNamespace(**config)
        
        global_config = GlobalConfig(config)
        
        dataloader = Dataloder(global_config)
    else:
        global_config = globalconfig
        if dataset_name:
            global_config.dataset_name = dataset_name
        if dataset_path:
            global_config.dataset_path = dataset_path
        if output_path:
            global_config.output_path = output_path
        if token_file:
            global_config.token_file = token_file
        if batch_size is None:
            batch_size = getattr(global_config, 'batch_size', 1)
        if num_processes is None:
            num_processes = getattr(global_config, 'num_processes', None)
        dataloader = Dataloder(global_config)

    token_file = token_file or getattr(global_config, 'token_file', DEFAULT_TOKEN_FILE)
    output_path = output_path or getattr(global_config, 'output_path', None)

    total_samples = len(dataloader.data)
    print(f"Total samples to process: {total_samples}")
    
    # 使用多进程批处理
    batch_size = batch_size or 1
    print(f"Using batch processing with batch_size={batch_size}")
    
    # 划分批次
    batches = split_into_batches(total_samples, batch_size)
    print(f"Split into {len(batches)} batches: {batches}")
    
    # 确定进程数
    if num_processes is None:
        num_processes = min(len(batches), mp.cpu_count())
    
    print(f"Using {num_processes} processes")
    
    all_results = []
    all_start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 提交所有任务
        future_to_batch = {}
        for batch_id, batch_indices in enumerate(batches):
            future = executor.submit(process_batch, batch_indices, global_config, token_file, batch_id)
            future_to_batch[future] = batch_id
        
        # 收集结果
        batch_results_list = []
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                batch_results = future.result()
                batch_results_list.append((batch_id, batch_results))
                print(f"Batch {batch_id} completed successfully")
            except Exception as exc:
                print(f"Batch {batch_id} generated an exception: {exc}")
                # 可以选择继续处理其他批次或者抛出异常
                raise exc
        
        # 按批次ID排序并合并结果
        batch_results_list.sort(key=lambda x: x[0])
        for batch_id, batch_results in batch_results_list:
            all_results.extend(batch_results)
    
    all_end_time = time.time()
    print(f"所有 {total_samples} 个样本处理完成，使用了 {len(batches)} 个批次，时间消耗: {all_end_time - all_start_time:.2f}秒")
    # # 按sample_id排序以保持原有顺序
    # all_results.sort(key=lambda x: int(x["sample_id"].replace("sample", "")))
    
    # 保存结果
    if output_path:
        import json
        ensure_parent_dir(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"All sample results have been saved to: {output_path}")
    
    # 合并token追踪文件
    ensure_parent_dir(token_file)
    merge_token_files(token_file, len(batches))
    
    return all_results

def merge_token_files(base_token_file, num_batches):
    """合并多个批次的token追踪文件"""
    import json
    
    merged_data = {}
    
    for batch_id in range(num_batches):
        batch_token_file = f"{base_token_file.replace('.json', '')}_batch_{batch_id}.json"
        try:
            with open(batch_token_file, 'r') as f:
                batch_data = json.load(f)
                # 合并数据，可以根据需要调整合并逻辑
                for key, value in batch_data.items():
                    if key in merged_data:
                        if isinstance(value, (int, float)):
                            merged_data[key] += value
                        elif isinstance(value, list):
                            merged_data[key].extend(value)
                        # 其他类型根据需要处理
                    else:
                        merged_data[key] = value
 
        except FileNotFoundError:
            print(f"Warning: Token file {batch_token_file} not found")
    
    # 保存合并后的文件
    with open(base_token_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Token files merged into {base_token_file}")
