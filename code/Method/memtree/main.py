import argparse
import os
import sys
import time
from .config import globalconfig, load_config
from .utils import retrieve, generation
from .dataloader import Dataloder
from .structure import build_tree, save_tree, load_tree, MemTree
from .token_tracker import TokenTracker

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

def run_memtree(config_path=None, dataset_name=None, dataset_path=None, output_path=None,
                token_file=None):
    print("Running run_memtree...")
    
    """
    运行 MemoryTree 的主要函数
    
    Args:
        config_path: 配置文件路径
        dataset_name: 数据集名称
        output_path: 输出文件路径
    
    Returns:
        处理结果
    """
    
    # 如果提供了配置文件路径，重新加载配置
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

        config['config_path'] = config_path
        config = resolve_memtree_config_dict(config, config_path)
        output_path = config.get('output_path')
        token_file = config.get('token_file', token_file)
            
        config = SimpleNamespace(**config)
        
        global_config = GlobalConfig(config)
        
        # 创建数据加载器
        dataloader = Dataloder(global_config)
    else:
        global_config = globalconfig
        if dataset_name:
            global_config.dataset_name = dataset_name
        if dataset_path:
            global_config.dataset_path = dataset_path
        token_file = token_file or getattr(global_config, 'token_file', DEFAULT_TOKEN_FILE)
        output_path = output_path or getattr(global_config, 'output_path', None)
        dataloader = Dataloder(global_config)

    token_file = token_file or DEFAULT_TOKEN_FILE
    ensure_parent_dir(token_file)
    tracker = TokenTracker(output_file=token_file)
    tracker.patch_llm_api()

    # 存储所有样本的结果
    all_results = []
    start_time = time.time()  

    for i in range(len(dataloader.data)):
        # 为每个样本创建独立的数据库配置
        print(f"Processing sample {i}, creating independent database...")
        
        # 创建当前样本的独立配置
        sample_config = create_sample_config(global_config, i)
        
        # 更新数据加载器的配置
        dataloader.update_config(sample_config)
        
        # 临时更新全局配置，以便其他模块能够使用新的数据库设置
        update_global_config(sample_config)
        
        # 构建树结构
        print("building tree for index:", i)
        with tracker.stage(f"Sample {i}"):
            # tree = build_tree(dataloader, i, tracker)

            tree = load_tree(globalconfig.save_path, i)
            questions, sessions = dataloader.data[i]
            
            if tree is None:
                tree = MemTree("")
                root_id = id(tree.root)
                all_sessions = []
                for session_id, session in sessions.items():
                    with tracker.stage(f"Session {session_id}"):
                        all_sessions.extend(session)
                        # break # 目前单session测试
                        dial_id = 0
                        for dial in session:
                            with tracker.stage(f"Dialog {dial_id}"):
                                tree.add_node(dial, root_id)
                            dial_id += 1

                save_tree(tree, globalconfig.save_path, i)
        
        # 获取问题数据
        questions = dataloader.data[i][0]
    
        # 检索相关内容
        print("retrieving data...")
        retrieve_result = retrieve(questions, i, token_file)
    
        print("检索结果:")
        print(retrieve_result)
    
        # 生成回答
        result = generation(tree, retrieve_result, i, token_file)
    
        print("生成结果:")
        for question, context, answer in result:
            print(f"问题: {question}")
            print(f"答案: {answer}")
            print("-" * 50)
        
        # # 如果指定了输出路径，保存结果
        # if output_path:
        #     import json
        #     output_data = []
        #     for question, context, answer in result:
        #         output_data.append({
        #             "question": question,
        #             "context": context,
        #             "answer": answer
        #         })
            
        #     # 为每个样本创建独立的输出文件
        #     base_name, ext = os.path.splitext(output_path)
        #     sample_output_path = f"{base_name}_sample_{i}{ext}"
            
        #     with open(sample_output_path, 'w', encoding='utf-8') as f:
        #         json.dump(output_data, f, ensure_ascii=False, indent=2)
        #     print(f"结果已保存到: {sample_output_path}")
        # else:
        #     # 如果没有指定输出路径，使用配置中的默认路径
        #     import json
        #     output_data = []
        #     for question, context, answer in result:
        #         output_data.append({
        #             "question": question,
        #             "context": context,
        #             "answer": answer
        #         })
            
        #     with open(sample_config.save_path, 'w', encoding='utf-8') as f:
        #         json.dump(output_data, f, ensure_ascii=False, indent=2)
        #     print(f"结果已保存到: {sample_config.save_path}")
        
        # 将当前样本的结果转换为期望的格式
        qa_list = []
        for question, context, answer in result:
            # 如果question是字符串，直接使用；如果是字典对象，提取问题文本和答案
            if isinstance(question, dict):
                question_text = question.get("question", "")
                expected_answer = question.get("answer", "")
                category = question.get("category", "")
            else:
                question_text = str(question)
                expected_answer = ""
            
            # 将context按换行符分割为retrieved列表
            retrieved = context.split('\n\n') if context else []
            
            qa_item = {
                "question": question_text,
                "answer": expected_answer,
                "category": category,
                "response": answer,
                "retrieved": retrieved
            }
            qa_list.append(qa_item)
        
        # 按照期望格式添加到结果中
        result = {
            "sample_id": dataloader.sample_ids[i],  # 从sample1开始
            "qa": qa_list
        }
        all_results.append(result)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"所有样本处理完成，总耗时: {elapsed_time:.2f}秒")
    if output_path:
        import json
        ensure_parent_dir(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"所有样本的结果已保存到: {output_path}")
        
    # 恢复原始全局配置（如果需要的话）
    update_global_config(global_config)
    
    print(f"所有 {len(dataloader.data)} 个样本处理完成，每个样本都有独立的数据库")
    
    return all_results # 返回所有样本的结果

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='MemoryTree 主程序')
#     parser.add_argument('--config', type=str, help='配置文件路径')
#     parser.add_argument('--dataset', type=str, help='数据集名称')
#     parser.add_argument('--output', type=str, help='输出文件路径')

#     args = parser.parse_args()
    
#     # 如果没有提供任何参数，使用默认方式运行
#     if not any(vars(args).values()):
#         # 原始的运行方式
#         from .dataloader import dataloader
        
#         tree = build_tree(dataloader,0)
#         questions = dataloader.data[0][0]
#         # 筛选D1的
#         d1_query = []
#         for item in questions:  
#             if item["evidence"] and "answer" in item:
#                 if "D1:" in item["evidence"][0]:
#                     try:
#                         d1_query.append((item["question"], item["answer"]))
#                     except:
#                         breakpoint()
        
#         questions = list(map(lambda x: x[0], d1_query))
        
#         retrieve_result = retrieve(questions)
        
#         print(retrieve_result)
        
#         result = generation(tree, retrieve_result)
#         print(result)
#     else:
#         # 使用参数化的方式运行
#         run_memtree(
#             config_path=args.config,
#             dataset_name=args.dataset,
#             output_path=args.output
#         )
