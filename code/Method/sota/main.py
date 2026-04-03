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
from .token_tracker import TokenTracker

DEFAULT_LLM_API_KEY = "empty"
DEFAULT_LLM_BASE_URL = "http://localhost:8001/v1"
DEFAULT_LLM_MODEL = "/home/docker/LLaMA-Factory/output/qwen2_5_lora_sft"
DEFAULT_EMBEDDING_MODEL_NAME = "minilm"

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
    sample_db_name = os.path.join(data_dir, f'{clean_str(base_config.embedding_model_name).replace(" ", "")}_sample_{sample_index}_{base_config.vdb_name}')
    
    # Update the configuration.
    sample_config.db_name = sample_db_name  # Milvus database path
    sample_config.collection_name = f"{base_config.collection_name}_sample_{sample_index}"
    sample_config.save_path = os.path.join(data_dir, f"{base_config.save_name}")  # Tree file path

    # Create a new Milvus client and collection.
    sample_config.client = MilvusClient(sample_config.db_name)
    create_collections(sample_config.client, sample_config.collection_name, base_config.dimension)
    
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

def simple_sample(conv_data, data_storage_path, tree, sample_config, i, tracker):
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
            short_term_capacity=7,
            tree=tree,
            segment_threshold=0.5
        )
    
    # Process each session and add memories.
    session_count = 0
    with tracker.stage(f"Sample {i}"):
        for key in conversation.keys():
            if key.startswith('session_') and not key.endswith('_date_time'):
                print(f"   📅 Processing {key}...")
                session_data = conversation[key]
                date_time = conversation.get(f"{key}_date_time", "")
                session_count += 1

                with tracker.stage(f"Session {session_count}"):
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
                        with tracker.stage(f"Dialog {j//2}"):
                            memo.add_memory(
                                speaker_a,
                                speaker_b,
                                speaker_a_input,
                                speaker_b_input,
                                timestamp=parse_datetime_string(date_time)
                            )
                print(f"   ✅ Added memory: {speaker_a} & {speaker_b}, session: {session_count}")
    
    save_tree(tree, sample_config.save_path, i)
    
    qa_pairs = conv_data['qa']
    qa_results = []
    for qa in qa_pairs:
        print(f"   ❓ Answering question: {qa['question']}")
        question = qa["question"]
        original_answer = qa.get("answer", "")
        category = qa["category"]
        retrieved, system_answer = memo.get_response(query=question, mode='split', speaker_a=speaker_a, speaker_b=speaker_b)
        qa_result = {
            "question": question,
            "answer": original_answer,
            "category": category,
            "response": system_answer,
            "retrieved": retrieved
            }
        qa_results.append(qa_result)
        print(f"   ✅ Question answered.")
    sample_qa_result = {
        "sample_id": sample_id,
        "qa": qa_results
        }
    return sample_qa_result

def run_sota(
    dataset_path,
    output_path,
    memory_path,
    config_path=None,
    llm_model=DEFAULT_LLM_MODEL,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
):
    tracker = TokenTracker(output_file=output_path.replace(".json", "_tokens.json"))
    tracker.patch_llm_api()

    llm_model = llm_model or DEFAULT_LLM_MODEL
    llm_api_key = llm_api_key or DEFAULT_LLM_API_KEY
    llm_base_url = llm_base_url or DEFAULT_LLM_BASE_URL
    embedding_model_name = embedding_model_name or DEFAULT_EMBEDDING_MODEL_NAME

    if config_path:
        from types import SimpleNamespace
        import yaml
        
        with open(config_path) as f:
            config = yaml.safe_load(f)

        config["dataset_path"] = dataset_path
        config["output_path"] = output_path
        config["memory_path"] = memory_path
        config["llm_model"] = llm_model
        config["llm_api_key"] = llm_api_key
        config["llm_base_url"] = llm_base_url
        config["embedding_model_name"] = embedding_model_name
            
        config = SimpleNamespace(**config)
        
        from .config import GlobalConfig
        global_config = GlobalConfig(config)
        
    else:
        print("No config_path provided.")
        return
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    total_samples = len(dataset)
    print(f"Processing {total_samples} samples sequentially...")
    
    all_results = []
    
    # Resume from existing results if available.
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        print(f"Found {len(all_results)} already processed samples, resuming...")
    
    all_start_time = time.time()
    
    # Process each sample sequentially.
    for i in range(len(all_results), total_samples):
        conv_data = dataset[i]
        
        # Create an isolated configuration for the current sample.
        sample_config = create_sample_config(global_config, i)
        
        # Temporarily update the global configuration.
        update_global_config(sample_config)
        
        # Build or load the tree structure.
        tree = load_tree(sample_config.save_path, i)
        if tree is None:
            tree = MemTree("", api_key=sample_config.llm_api_key, base_url=sample_config.llm_base_url, model=sample_config.llm_model, mode='default')

        print(f"Processing sample {i}/{total_samples}")
        
        data_storage_path = os.path.join(memory_path, f"mem_data_sample_{i}")

        sample_qa_result = simple_sample(conv_data, data_storage_path, tree, sample_config, i, tracker)
        all_results.append(sample_qa_result)
        
        # Save results after each sample.
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    all_end_time = time.time()
    print(f"All {total_samples} samples have been processed. Elapsed time: {all_end_time - all_start_time:.2f} seconds")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    return all_results

if __name__ == "__main__":
    run_sota(dataset_path="/home/docker/IndepthMem/Dataset/LOCOMO/locomo1demo.json", 
             memory_path="/home/docker/IndepthMem/Result/LOCOMO/sota/test/mem_data",
             output_path="/home/docker/IndepthMem/Result/LOCOMO/sota/test/result.json",
             config_path="/home/docker/IndepthMem/Config/sota.yaml")
