import argparse
import contextlib
import json
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
from ..dataset_hygiene import raw_result_path

mp.set_start_method('spawn', force=True)

DEFAULT_TOKEN_FILE = None


class NullTracker:
    def stage(self, name):
        return contextlib.nullcontext()


def ensure_parent_dir(path):
    if path:
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)


def create_sample_config(base_config, sample_index):
    """
    为指定的样本创建独立的配置
    """
    from types import SimpleNamespace
    from pymilvus import MilvusClient
    from .config import clean_str, create_collections

    config_dict = {}
    for attr in dir(base_config):
        if not attr.startswith('_'):
            config_dict[attr] = getattr(base_config, attr)

    sample_config = SimpleNamespace(**config_dict)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data", base_config.dataset_name)
    sample_db_name = os.path.join(
        data_dir,
        f'{clean_str(base_config.embedding_model_name).replace(" ", "")}_sample_{sample_index}_{base_config.vdb_name}'
    )

    sample_config.db_name = sample_db_name
    sample_config.collection_name = f"{base_config.collection_name}_sample_{sample_index}"
    sample_config.save_path = os.path.join(data_dir, f"{base_config.save_name}")

    sample_config.client = MilvusClient(sample_config.db_name)
    create_collections(sample_config.client, sample_config.collection_name, base_config.dimension)
    sample_config.model = base_config.model

    print(f"Created independent database for sample {sample_index}: {sample_config.db_name}")
    return sample_config


def update_global_config(new_config):
    """
    临时更新全局配置以供其他模块使用
    """
    from . import config

    if config.globalconfig is None:
        print("Warning: globalconfig is None, cannot update")
        return

    core_attrs = [
        'db_name', 'collection_name', 'client', 'model', 'save_path',
        'dataset_name', 'dimension', 'embedding_model_name', 'embedding_device', 'vdb_name',
        'embedding_batch_size', 'llm_parallel_nums', 'answer_parallel_nums',
        'base_threshold', 'rate', 'max_depth', 'top_k_retrieve', 'retrieve_top_ks',
        'llm_base_url', 'llm_api_key', 'llm_model'
    ]

    updated_attrs = []
    for attr in core_attrs:
        if hasattr(new_config, attr):
            setattr(config.globalconfig, attr, getattr(new_config, attr))
            updated_attrs.append(attr)

    print(f"Updated global config attributes: {updated_attrs}")
    if hasattr(new_config, 'db_name'):
        print(f"Updated global config with new database: {new_config.db_name}")

    if hasattr(config.globalconfig, 'collection_name'):
        print(f"Verification: globalconfig.collection_name = {config.globalconfig.collection_name}")
    else:
        print("Warning: collection_name still missing after update")


def split_into_batches(total_samples, batch_size):
    batches = []
    for i in range(0, total_samples, batch_size):
        batch = list(range(i, min(i + batch_size, total_samples)))
        batches.append(batch)
    return batches


def parse_retrieve_top_ks(retrieve_top_ks):
    if retrieve_top_ks is None:
        return [10]
    if isinstance(retrieve_top_ks, int):
        values = [retrieve_top_ks]
    elif isinstance(retrieve_top_ks, str):
        values = [part.strip() for part in retrieve_top_ks.split(',') if part.strip()]
    else:
        values = list(retrieve_top_ks)

    top_ks = []
    for value in values:
        k = int(value)
        if k <= 0:
            raise ValueError(f"retrieve top-k must be positive, got {value}")
        if k not in top_ks:
            top_ks.append(k)
    return top_ks or [10]


def build_topk_output_paths(output_path, retrieve_top_ks):
    if len(retrieve_top_ks) == 1 and output_path and output_path.endswith('.json'):
        return {retrieve_top_ks[0]: raw_result_path(output_path)}
    output_root = output_path if output_path and not output_path.endswith('.json') else os.path.dirname(output_path or '.')
    return {
        k: raw_result_path(os.path.join(output_root, f"top_k_{k}"))
        for k in retrieve_top_ks
    }


def load_existing_results_by_k(output_paths):
    results_by_k = {}
    processed_by_k = {}
    for k, output_file in output_paths.items():
        results = []
        processed = set()
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                if isinstance(existing_results, list):
                    results = existing_results
                    processed = {result.get('sample_id') for result in results if result.get('sample_id')}
                    print(f"Existing top_k={k} results detected: {len(processed)} processed samples.")
            except Exception as e:
                print(f"Error while reading {output_file}: {e}. Restarting that top-k from scratch.")
        else:
            print(f"No existing results file for top_k={k}. Starting from scratch.")
        results_by_k[k] = results
        processed_by_k[k] = processed
    return results_by_k, processed_by_k


def save_results(output_file, results):
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def sort_results_by_sample_order(results, sample_order):
    return sorted(results, key=lambda item: sample_order.get(item.get('sample_id'), 10**9))


def merge_results_by_k(base_results_by_k, new_results_by_k, sample_order):
    for top_k, new_results in new_results_by_k.items():
        existing = {item.get('sample_id'): item for item in base_results_by_k.setdefault(top_k, [])}
        for item in new_results:
            sample_id = item.get('sample_id')
            if sample_id:
                existing[sample_id] = item
        base_results_by_k[top_k] = sort_results_by_sample_order(list(existing.values()), sample_order)


def save_all_outputs(output_paths, results_by_k, manifest_path=None):
    manifest = {"retrieve_top_ks": sorted(output_paths.keys()), "outputs": {}}
    for top_k, output_file in output_paths.items():
        save_results(output_file, results_by_k.get(top_k, []))
        latencies = [
            qa.get('retrieval_latency_ms')
            for sample in results_by_k.get(top_k, [])
            for qa in sample.get('qa', [])
            if isinstance(qa.get('retrieval_latency_ms'), (int, float))
        ]
        manifest["outputs"][str(top_k)] = {
            "result_file": output_file,
            "sample_count": len(results_by_k.get(top_k, [])),
            "qa_count": sum(len(sample.get('qa', [])) for sample in results_by_k.get(top_k, [])),
            "average_retrieval_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        }
    if manifest_path:
        save_results(manifest_path, manifest)


def format_qa_results(generated_results, retrieve_top_k):
    qa_list = []
    for item in generated_results:
        if len(item) == 4:
            question, context, answer, retrieval_latency_ms = item
        else:
            question, context, answer = item
            retrieval_latency_ms = 0.0

        if isinstance(question, dict):
            question_text = question.get("question", "")
            expected_answer = question.get("answer", "")
            category = question.get("category", "")
        else:
            question_text = str(question)
            expected_answer = ""
            category = ""

        retrieved = context.split('\n\n') if context else []
        qa_list.append({
            "question": question_text,
            "answer": expected_answer,
            "category": category,
            "response": answer,
            "retrieved": retrieved,
            "retrieved_count": len(retrieved),
            "retrieval_top_k": retrieve_top_k,
            "retrieval_latency_ms": retrieval_latency_ms,
            "search_duration_ms": retrieval_latency_ms,
        })
    return qa_list


def process_batch(batch_indices, global_config, token_file, batch_id, retrieve_top_ks, processed_by_k, track_tokens=True):
    """
    处理单个批次的样本。每个样本只建立一次索引，然后复用最大 top-k 的检索结果生成多个 top-k 输出。
    """
    start_time = time.time()
    from .dataloader import Dataloder
    from .token_tracker import TokenTracker

    if track_tokens:
        batch_token_file = f"{token_file.replace('.json', '')}_batch_{batch_id}.json"
        ensure_parent_dir(batch_token_file)
        tracker = TokenTracker(output_file=batch_token_file)
        tracker.patch_llm_api()
    else:
        batch_token_file = token_file
        tracker = NullTracker()
        print(f"Batch {batch_id}: token tracking disabled.")

    print(f"Batch {batch_id}: Processing samples {batch_indices}")

    dataloader = Dataloder(global_config)
    batch_results_by_k = {k: [] for k in retrieve_top_ks}
    max_retrieve_top_k = max(retrieve_top_ks)

    for i in batch_indices:
        sample_id = dataloader.sample_ids[i]
        if all(sample_id in processed_by_k.get(k, set()) for k in retrieve_top_ks):
            print(f"Batch {batch_id}: Sample {sample_id} already processed for all top-k values. Skipping.")
            continue

        print(f"Batch {batch_id}: Processing sample {i} ({sample_id})")

        sample_config = create_sample_config(global_config, i)
        dataloader.update_config(sample_config)
        update_global_config(sample_config)

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

        questions = dataloader.data[i][0]

        print(f"Batch {batch_id}: retrieving data for sample {i} with max top_k={max_retrieve_top_k}")
        retrieve_result = retrieve(questions, i, batch_token_file, top_k=max_retrieve_top_k)

        for retrieve_top_k in retrieve_top_ks:
            if sample_id in processed_by_k.get(retrieve_top_k, set()):
                continue

            print(f"Batch {batch_id}: generating answers for sample {i}, top_k={retrieve_top_k}")
            truncated_retrieve_result = [
                (question, context_ids[:retrieve_top_k], retrieval_latency_ms)
                for question, context_ids, retrieval_latency_ms in retrieve_result
            ]
            generated = generation(tree, truncated_retrieve_result, i, batch_token_file)
            batch_results_by_k[retrieve_top_k].append({
                "sample_id": sample_id,
                "qa": format_qa_results(generated, retrieve_top_k),
            })
            print(f"Batch {batch_id}: Completed sample {sample_id} for top_k={retrieve_top_k}")

    end_time = time.time()
    print(f"Batch {batch_id}: Completed samples {batch_indices}, elapsed: {end_time - start_time:.2f}s")
    return batch_results_by_k


def run_memtree(config_path=None, dataset_name=None, dataset_path=None, output_path=None,
                token_file=None, batch_size=None, num_processes=None,
                retrieve_top_ks=None, track_tokens=True):
    """
    运行 MemoryTree。支持一次建索引后复用检索结果生成多个 retrieve top-k 输出。
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
        if retrieve_top_ks is not None:
            config['retrieve_top_ks'] = retrieve_top_ks
        if track_tokens is not None:
            config['track_tokens'] = track_tokens

        config['config_path'] = config_path
        config = resolve_memtree_config_dict(config, config_path)
        output_path = config.get('output_path')
        token_file = config.get('token_file', token_file)
        batch_size = config.get('batch_size', 1)
        num_processes = config.get('num_processes', num_processes)
        retrieve_top_ks = config.get('retrieve_top_ks', retrieve_top_ks)
        track_tokens = bool(config.get('track_tokens', track_tokens))

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
        if retrieve_top_ks is None:
            retrieve_top_ks = getattr(global_config, 'retrieve_top_ks', None)
        track_tokens = bool(getattr(global_config, 'track_tokens', track_tokens))
        dataloader = Dataloder(global_config)

    retrieve_top_ks = parse_retrieve_top_ks(retrieve_top_ks)
    if not output_path:
        output_path = getattr(global_config, 'output_path', None)
    if not output_path:
        raise ValueError('MemoryTree requires output_path in config or CLI args.')

    token_file = token_file or getattr(global_config, 'token_file', None)
    if not token_file:
        output_root = output_path if not output_path.endswith('.json') else os.path.dirname(output_path)
        token_file = os.path.join(os.path.abspath(output_root or '.'), 'token_tracker.json')
    if track_tokens:
        ensure_parent_dir(token_file)

    output_paths = build_topk_output_paths(output_path, retrieve_top_ks)
    output_root = output_path if not output_path.endswith('.json') else os.path.dirname(output_path)
    manifest_path = os.path.join(output_root, 'memtree_run_manifest.json')
    for top_k, output_file in output_paths.items():
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        print(f"top_k={top_k} results will be saved to {output_file}")

    results_by_k, processed_by_k = load_existing_results_by_k(output_paths)

    total_samples = len(dataloader.data)
    sample_order = {sample_id: idx for idx, sample_id in enumerate(dataloader.sample_ids)}
    pending_indices = [
        idx for idx, sample_id in enumerate(dataloader.sample_ids)
        if not all(sample_id in processed_by_k[k] for k in retrieve_top_ks)
    ]

    print(f"Total samples in dataset: {total_samples}")
    print(f"Pending samples to process: {len(pending_indices)}")
    if not pending_indices:
        save_all_outputs(output_paths, results_by_k, manifest_path)
        return results_by_k

    batch_size = batch_size or 1
    batches = [pending_indices[i:i + batch_size] for i in range(0, len(pending_indices), batch_size)]
    print(f"Using batch_size={batch_size}; split pending samples into {len(batches)} batches: {batches}")

    if num_processes is None:
        num_processes = min(len(batches), mp.cpu_count())
    num_processes = max(1, int(num_processes))
    print(f"Using {num_processes} processes")

    all_start_time = time.time()

    if num_processes == 1:
        for batch_id, batch_indices in enumerate(batches):
            batch_results = process_batch(
                batch_indices, global_config, token_file, batch_id,
                retrieve_top_ks, processed_by_k, track_tokens=track_tokens,
            )
            merge_results_by_k(results_by_k, batch_results, sample_order)
            for k, sample_results in batch_results.items():
                processed_by_k[k].update(item.get('sample_id') for item in sample_results if item.get('sample_id'))
            save_all_outputs(output_paths, results_by_k, manifest_path)
            print(f"Batch {batch_id} merged and saved.")
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_to_batch = {}
            for batch_id, batch_indices in enumerate(batches):
                future = executor.submit(
                    process_batch, batch_indices, global_config, token_file, batch_id,
                    retrieve_top_ks, processed_by_k, track_tokens,
                )
                future_to_batch[future] = batch_id

            for future in concurrent.futures.as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_results = future.result()
                    merge_results_by_k(results_by_k, batch_results, sample_order)
                    for k, sample_results in batch_results.items():
                        processed_by_k[k].update(item.get('sample_id') for item in sample_results if item.get('sample_id'))
                    save_all_outputs(output_paths, results_by_k, manifest_path)
                    print(f"Batch {batch_id} completed, merged, and saved.")
                except Exception as exc:
                    print(f"Batch {batch_id} generated an exception: {exc}")
                    raise exc

    elapsed = time.time() - all_start_time
    print(f"所有 {len(pending_indices)} 个待处理样本完成，时间消耗: {elapsed:.2f}秒")
    save_all_outputs(output_paths, results_by_k, manifest_path)
    print(f"Run manifest saved to: {manifest_path}")

    if track_tokens:
        merge_token_files(token_file, len(batches))
    else:
        print("Token tracking disabled; no token files merged.")

    return results_by_k


def merge_token_files(base_token_file, num_batches):
    """合并多个批次的token追踪文件"""
    merged_data = {}

    for batch_id in range(num_batches):
        batch_token_file = f"{base_token_file.replace('.json', '')}_batch_{batch_id}.json"
        try:
            with open(batch_token_file, 'r') as f:
                batch_data = json.load(f)
                for key, value in batch_data.items():
                    if key in merged_data:
                        if isinstance(value, (int, float)):
                            merged_data[key] += value
                        elif isinstance(value, list):
                            merged_data[key].extend(value)
                    else:
                        merged_data[key] = value
        except FileNotFoundError:
            print(f"Warning: Token file {batch_token_file} not found")

    with open(base_token_file, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"Token files merged into {base_token_file}")
