from .memoryos import Memoryos
import os
import json
import re
import time
from .config import globalconfig, clean_str, create_collections
# from .dataloader import Dataloder
from .structure import save_tree, load_tree, MemTree
from types import SimpleNamespace
from pymilvus import MilvusClient
import concurrent.futures
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

DEFAULT_LLM_API_KEY = os.getenv("ABLATION_LLM_API_KEY", "EMPTY")
DEFAULT_LLM_BASE_URL = os.getenv("ABLATION_LLM_BASE_URLS")
DEFAULT_LLM_MODEL = os.getenv("ABLATION_LLM_MODEL")
DEFAULT_EMBEDDING_MODEL_NAME = os.getenv("ABLATION_EMBEDDING_MODEL")

def create_sample_config(base_config, sample_index):
    """
    Create an isolated configuration for a specific sample.

    Args:
        base_config: The base configuration object.
        sample_index: The sample index.

    Returns:
        A new configuration object with an independent database setup.
    """

    
    # Copy all attributes from the base configuration.
    config_dict = {}
    for attr in dir(base_config):
        if not attr.startswith('_'):
            config_dict[attr] = getattr(base_config, attr)
    
    # Create a new configuration object.
    sample_config = SimpleNamespace(**config_dict)
    
    # Create an independent database name.
    output_dir = os.path.dirname(base_config.output_path)
    data_dir = os.path.join(output_dir, "database")
    os.makedirs(data_dir, exist_ok=True)
    embedding_storage_name = getattr(base_config, "embedding_storage_name", base_config.embedding_model_name)
    sample_db_name = os.path.join(data_dir, f'{clean_str(embedding_storage_name).replace(" ", "")}_sample_{sample_index}_{base_config.vdb_name}')
    
    # Update the configuration.
    sample_config.db_name = sample_db_name  # Milvus database path
    sample_config.collection_name = f"{base_config.collection_name}_sample_{sample_index}"
    sample_config.save_path = os.path.join(data_dir, f"{base_config.save_name}")  # Tree file path
    graph_save_name = getattr(base_config, "graph_save_name", "memory_graph.json")
    sample_config.graph_save_path = os.path.join(data_dir, f"sample_{sample_index}_{graph_save_name}")

    # Milvus Lite can race while several spawned processes initialise its mmap
    # manager simultaneously. Retry with bounded exponential backoff.
    milvus_attempts = int(getattr(base_config, "milvus_init_attempts", 5))
    for attempt in range(1, milvus_attempts + 1):
        client = None
        try:
            client = MilvusClient(sample_config.db_name)
            create_collections(
                client, sample_config.collection_name, base_config.dimension
            )
            sample_config.client = client
            break
        except Exception as exc:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            if attempt >= milvus_attempts:
                raise
            delay = min(5.0, 0.5 * (2 ** (attempt - 1))) + 0.05 * (sample_index % 10)
            print(
                f"Milvus init attempt {attempt}/{milvus_attempts} failed for "
                f"sample {sample_index}: {exc}. Retrying in {delay:.2f}s"
            )
            time.sleep(delay)
    
    # Reuse the same embedding model without reloading it.
    sample_config.model = base_config.model
    
    print(f"Created independent database for sample {sample_index}: {sample_config.db_name}")
    
    return sample_config

def update_global_config(new_config):
    """
    Temporarily update the global configuration for other modules.

    Args:
        new_config: The new configuration object.
    """
    # Import and update the global configuration.
    from . import config
    
    if config.globalconfig is None:
        config.globalconfig = new_config
        print("Initialized global config")
    
    # Update only the core attributes used by the runtime.
    core_attrs = ['db_name', 'collection_name', 'client', 'save_path']
    
    for attr in core_attrs:
        if hasattr(new_config, attr):
            setattr(config.globalconfig, attr, getattr(new_config, attr))
    
    print(f"Updated global config with new database: {new_config.db_name}")
    
    # Verify that the update was applied successfully.
    if hasattr(config.globalconfig, 'collection_name'):
        print(f"Verification: globalconfig.collection_name = {config.globalconfig.collection_name}")
    else:
        print("Warning: collection_name still missing after update")

def parse_datetime_string(datetime_str):
    """
    Parse datetime string to datetime object.
    Automatically detects and handles two formats:
    - LOCOMO Pattern: "1:56 pm on 8 May, 2023"
    - LONGMEMEVAL Pattern: "2023/05/20 (Sat) 02:21"
    """
    # Try LOCOMO Pattern first: "1:56 pm on 8 May, 2023"
    locomo_pattern = r'(\d{1,2}):(\d{2})\s+(am|pm)\s+on\s+(\d{1,2})\s+(\w+),\s+(\d{4})'
    locomo_match = re.match(locomo_pattern, datetime_str)
    
    if locomo_match:
        hour, minute, ampm, day, month_name, year = locomo_match.groups()
        
        # Convert to 24-hour format
        hour = int(hour)
        if ampm.lower() == 'pm' and hour != 12:
            hour += 12
        elif ampm.lower() == 'am' and hour == 12:
            hour = 0
        
        # Month name to number mapping
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        
        month = month_map.get(month_name)
        if not month:
            raise ValueError(f"Unknown month: {month_name}")
        
        return f"{int(year)}-{month}-{int(day):02d} {hour:02d}:{int(minute):02d}:00"
    
    # Try LONGMEMEVAL Pattern: "2023/05/20 (Sat) 02:21"
    longmemeval_pattern = r'(\d{4})/(\d{2})/(\d{2})\s+\([A-Za-z]{3}\)\s+(\d{2}):(\d{2})'
    longmemeval_match = re.match(longmemeval_pattern, datetime_str)
    
    if longmemeval_match:
        year, month, day, hour, minute = longmemeval_match.groups()
        
        # Convert strings to integers
        year = int(year)
        month = int(month)
        day = int(day)
        hour = int(hour)
        minute = int(minute)
        
        return f"{year}-{month}-{day:02d} {hour:02d}:{minute:02d}:00"
    
    # If no pattern matches, raise an error
    raise ValueError(f"Cannot parse datetime string: {datetime_str}. ")

def simple_sample(conv_data, data_storage_path, tree, sample_config, i):
    # Extract conversation metadata.
    conversation = conv_data['conversation']
    speaker_a = conversation['speaker_a']
    speaker_b = conversation['speaker_b']
    sample_id = conv_data['sample_id']
    memo = Memoryos(
            user_id=f"{sample_id}",
            openai_api_key=sample_config.llm_api_key,
            openai_base_url=sample_config.llm_base_url,
            data_storage_path=data_storage_path,
            llm_model=sample_config.llm_model,
            short_term_capacity=int(getattr(sample_config, "short_term_capacity", 7)),
            tree=tree,
            segment_threshold=float(getattr(sample_config, "segment_threshold", 0.5)),
            memory_granularity=sample_config.memory_granularity,
            mid_term_structure=sample_config.mid_term_structure,
            graph_path=sample_config.graph_save_path,
            graph_options={
                "context_window": getattr(sample_config, "graph_context_window", 3),
                "candidate_count": getattr(sample_config, "graph_candidate_count", 10),
                "dedupe_candidate_count": getattr(
                    sample_config, "graph_dedupe_candidate_count", 20
                ),
                "fuzzy_threshold": getattr(sample_config, "graph_fuzzy_threshold", 0.90),
                "search_hops": getattr(sample_config, "graph_search_hops", 1),
                "use_llm_dedup": getattr(sample_config, "graph_use_llm_dedup", True),
                "use_edge_dedup": getattr(sample_config, "graph_use_edge_dedup", True),
            },
            top_k_retrieve=sample_config.top_k_retrieve,
            graph_only_retrieval=bool(
                getattr(sample_config, "graph_only_retrieval", False)
            ),
        )
    
    # Process each session and add memories.
    session_count = 0
    for key in conversation.keys():
        if key.startswith('session_') and not key.endswith('_date_time'):
            print(f"   📅 Processing {key}...")
            session_data = conversation[key]
            date_time = conversation.get(f"{key}_date_time", "")
            session_count += 1

            # Convert each dialogue pair in the session into memory entries.
            for j in range(0, len(session_data) - 1, 2):
                current_message = session_data[j]
                next_message = session_data[j + 1]
                current_speaker = current_message['speaker']
                current_text = current_message['text']
                next_speaker = next_message['speaker']
                next_text = next_message['text']
                
                # Use speaker_a as user input and speaker_b as agent response.
                if current_speaker == speaker_a:
                    speaker_a_input = current_text
                    speaker_b_input = next_text
                else:
                    speaker_a_input = next_text
                    speaker_b_input = current_text
                
                # Add the memory entry.
                memo.add_memory(
                    speaker_a,
                    speaker_b,
                    speaker_a_input,
                    speaker_b_input,
                    timestamp=parse_datetime_string(date_time)
                )
            # print(f"   ✅ Added memory: {speaker_a} & {speaker_b}, session: {session_count}")
    
    if sample_config.mid_term_structure == "tree":
        save_tree(tree, sample_config.save_path, i)
    
    qa_pairs = conv_data['qa']
    qa_results = [None] * len(qa_pairs)

    def answer_question(index_and_qa):
        qa_index, qa = index_and_qa
        print(f"   ❓ Answering question: {qa['question']}")
        question = qa["question"]
        original_answer = qa.get("answer", "")
        category = qa["category"]
        retrieved, system_answer = memo.get_response(
            query=question,
            mode='split',
            speaker_a=speaker_a,
            speaker_b=speaker_b,
        )
        qa_result = {
            "question": question,
            "answer": original_answer,
            "category": category,
            "response": system_answer,
            "retrieved": retrieved,
        }
        print(f"   ✅ Question answered.")
        return qa_index, qa_result

    max_workers = int(getattr(sample_config, 'qa_parallel_nums', 1) or 1)
    max_workers = max(1, min(max_workers, len(qa_pairs))) if qa_pairs else 1
    if max_workers == 1:
        for indexed_qa in enumerate(qa_pairs):
            qa_index, qa_result = answer_question(indexed_qa)
            qa_results[qa_index] = qa_result
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(answer_question, indexed_qa)
                for indexed_qa in enumerate(qa_pairs)
            ]
            for future in concurrent.futures.as_completed(futures):
                qa_index, qa_result = future.result()
                qa_results[qa_index] = qa_result
    sample_qa_result = {
        "sample_id": sample_id,
        "qa": qa_results
        }
    return sample_qa_result

def split_into_batches(total_samples, num_processes):
    """Split sample indices into batches based on the process count."""
    batches = []
    # Compute the number of samples handled by each process.
    batch_size = (total_samples + num_processes - 1) // num_processes  # Round up.
    for i in range(0, total_samples, batch_size):
        batch = list(range(i, min(i + batch_size, total_samples)))
        batches.append(batch)
    return batches

def process_batch(batch_indices, batch_id, dataset_path, memory_path, output_path,
                  config_dict, llm_runtime):
    from .config import GlobalConfig
    from .utils import configure_llm_runtime
    from .token_tracker import get_token_usage, merge_token_usages, reset_token_usage, save_token_usage
    reset_token_usage()

    configure_llm_runtime(
        base_urls=config_dict["llm_base_url"],
        per_endpoint_concurrency=llm_runtime["per_endpoint_concurrency"],
        semaphores=llm_runtime["semaphores"],
        counter=llm_runtime["counter"],
        lock=llm_runtime["lock"],
    )
    global_config = GlobalConfig(SimpleNamespace(**config_dict))
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    batch_results = []
    data_storage_path = os.path.join(memory_path, f"mem_data_batch_{batch_id}")
    batch_output_path = output_path.replace('.json', f'_batch_{batch_id}.json')
    batch_token_path = output_path.replace('.json', f'_tokens_batch_{batch_id}.json')
    previous_token_usage = None
    if os.path.exists(batch_token_path):
        with open(batch_token_path, 'r', encoding='utf-8') as f:
            previous_token_usage = json.load(f)
    combined_token_usage = merge_token_usages([previous_token_usage, get_token_usage()])
    exist_sample = []
    if os.path.exists(batch_output_path):
        with open(batch_output_path, 'r', encoding='utf-8') as f:
            batch_results = json.load(f)
            exist_sample = batch_indices[:len(batch_results)]
            print(f"Batch {batch_id} already processed samples: {exist_sample}, skipping...")
    for i in batch_indices:
        if i in exist_sample:
            print(f"Skipping already processed sample {i} in batch {batch_id}")
            continue
        conv_data = dataset[i]

        # Create an isolated configuration for the current sample.
        sample_config = create_sample_config(global_config, i)

        # Temporarily update the global configuration.
        update_global_config(sample_config)

        # The graph loads itself from JSON; the existing tree path remains unchanged.
        tree = None
        if sample_config.mid_term_structure == "tree":
            tree = load_tree(sample_config.save_path, i)
            if tree is None:
                tree = MemTree("", api_key=sample_config.llm_api_key, base_url=sample_config.llm_base_url, model=sample_config.llm_model, mode='async')

        print(f"Processing sample {i} in batch {batch_id}")
        sample_qa_result = simple_sample(conv_data, data_storage_path, tree, sample_config, i)
        batch_results.append(sample_qa_result)
        with open(batch_output_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        combined_token_usage = merge_token_usages([previous_token_usage, get_token_usage()])
        save_token_usage(batch_token_path, combined_token_usage)
    return batch_results, combined_token_usage

def run_sota(
    dataset_path,
    output_path,
    memory_path,
    config_path=None,
    llm_model=DEFAULT_LLM_MODEL,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
    memory_granularity=None,
    mid_term_structure=None,
):

    llm_model = llm_model or DEFAULT_LLM_MODEL
    llm_api_key = llm_api_key or DEFAULT_LLM_API_KEY
    llm_base_url = llm_base_url or DEFAULT_LLM_BASE_URL
    embedding_model_name = embedding_model_name or DEFAULT_EMBEDDING_MODEL_NAME

    if not config_path:
        print("No config_path provided.")
        return

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if memory_granularity is not None:
        config["memory_granularity"] = memory_granularity
    if mid_term_structure is not None:
        config["mid_term_structure"] = mid_term_structure
    config.setdefault("memory_granularity", "segment")
    config.setdefault("mid_term_structure", "tree")
    if config["memory_granularity"] not in {"segment", "message"}:
        raise ValueError("memory_granularity must be 'segment' or 'message'")
    if config["mid_term_structure"] not in {"tree", "graph"}:
        raise ValueError("mid_term_structure must be 'tree' or 'graph'")
    print(
        f"Ablation: {config['memory_granularity']} extraction + "
        f"{config['mid_term_structure']} mid-term structure"
    )

    config["dataset_path"] = dataset_path
    config["output_path"] = output_path
    config["memory_path"] = memory_path
    config["llm_model"] = llm_model
    config["llm_api_key"] = llm_api_key
    config["llm_base_url"] = llm_base_url
    config["embedding_model_name"] = embedding_model_name

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Each sample remains isolated in its own process. LLM calls across all
    # processes share two endpoint semaphores, so the global cap is exact.
    total_samples = len(dataset)
    num_processes = min(int(config.get("num_processes", 1)), total_samples)
    base_urls = config["llm_base_url"]
    if not isinstance(base_urls, (list, tuple)):
        base_urls = [url.strip() for url in str(base_urls).split(",") if url.strip()]
    config["llm_base_url"] = list(base_urls)

    per_endpoint_concurrency = int(config.get("per_endpoint_concurrency", 16))
    configured_total = config.get("llm_parallel_nums")
    llm_parallel_nums = (
        int(configured_total)
        if configured_total is not None
        else per_endpoint_concurrency * len(base_urls)
    )
    if per_endpoint_concurrency * len(base_urls) != llm_parallel_nums:
        raise ValueError(
            "per_endpoint_concurrency * endpoint_count must equal "
            "llm_parallel_nums"
        )

    print(f"Using {num_processes} processes for {total_samples} samples")
    print(
        f"LLM concurrency: {llm_parallel_nums} total; "
        f"{per_endpoint_concurrency} per endpoint; endpoints={base_urls}"
    )

    batches = split_into_batches(total_samples, num_processes)
    print(f"Split into {len(batches)} batches: {batches}")

    all_results = []
    all_start_time = time.time()
    batch_results_list = []
    batch_token_usage_list = []

    with mp.Manager() as manager:
        llm_runtime = {
            "per_endpoint_concurrency": per_endpoint_concurrency,
            "semaphores": [
                manager.BoundedSemaphore(per_endpoint_concurrency)
                for _ in base_urls
            ],
            "counter": manager.Value("i", 0),
            "lock": manager.Lock(),
        }

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_to_batch = {}
            for batch_id, batch_indices in enumerate(batches):
                future = executor.submit(
                    process_batch,
                    batch_indices,
                    batch_id,
                    dataset_path,
                    memory_path,
                    output_path,
                    config,
                    llm_runtime,
                )
                future_to_batch[future] = batch_id

            for future in concurrent.futures.as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_results, batch_token_usage = future.result()
                    batch_results_list.append((batch_id, batch_results))
                    batch_token_usage_list.append(batch_token_usage)
                    print(f"Batch {batch_id} completed successfully")
                except Exception as exc:
                    print(f"Batch {batch_id} generated an exception: {exc}")
                    raise

    # Preserve dataset order independently of process completion order.
    batch_results_list.sort(key=lambda x: x[0])
    for batch_id, batch_results in batch_results_list:
        all_results.extend(batch_results)

    all_end_time = time.time()
    from .token_tracker import merge_token_usages, save_token_usage
    token_usage = merge_token_usages(batch_token_usage_list)
    token_usage["elapsed_seconds"] = round(all_end_time - all_start_time, 3)
    token_usage["memory_granularity"] = config["memory_granularity"]
    token_usage["mid_term_structure"] = config["mid_term_structure"]
    token_usage_path = os.path.join(os.path.dirname(output_path), "token_usage.json")
    save_token_usage(token_usage_path, token_usage)
    print(f"All {total_samples} samples have been processed using {len(batches)} batches. Elapsed time: {all_end_time - all_start_time:.2f} seconds")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    return all_results

if __name__ == "__main__":
    raise SystemExit("Use the unified entry point: python -m Method.ablation --help")
