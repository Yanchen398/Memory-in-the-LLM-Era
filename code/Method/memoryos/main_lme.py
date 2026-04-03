from memoryos import Memoryos
import os
import json
import concurrent.futures
import multiprocessing as mp
from .utils import DEFAULT_EMBEDDING_MODEL_NAME, DEFAULT_LLM_API_KEY, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL
mp.set_start_method('spawn', force=True)

def simple_sample(
    conv_data,
    data_storage_path,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    llm_model=DEFAULT_LLM_MODEL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
):
    # Read conversation metadata.
    conversation = conv_data['conversation']
    speaker_a = conversation['speaker_a']
    speaker_b = conversation['speaker_b']
    sample_id = conv_data['sample_id']
    memo = Memoryos(
            user_id=f"user_{sample_id}",
            openai_api_key=llm_api_key,
            openai_base_url=llm_base_url,
            data_storage_path=data_storage_path,
            llm_model=llm_model,
            embedding_model_name=embedding_model_name,
            assistant_id=f"assistant_{sample_id}",
            short_term_capacity=7,
            mid_term_heat_threshold=5,
        )
    
    # Process each session and add memories.
    session_count = 0
    for key in conversation.keys():
        if key.startswith('session_') and not key.endswith('_date_time'):
            print(f"Processing {key}...")
            session_data = conversation[key]
            session_count += 1
            
            # Convert session dialogs into memory entries.
            for j in range(0, len(session_data) - 1, 2):
                current_message = session_data[j]
                next_message = session_data[j + 1]
                
                current_speaker = current_message['speaker']
                current_text = current_message['text']
                next_speaker = next_message['speaker']
                next_text = next_message['text']
                
                # Use speaker_a as user_input and speaker_b as agent_response.
                if current_speaker == speaker_a:
                    user_input = current_text
                    agent_response = next_text
                else:
                    user_input = next_text
                    agent_response = current_text
                
                # Add the memory entry.
                memo.add_memory(
                    user_input=user_input,
                    agent_response=agent_response
                )
                print(f"Added memory for {speaker_a} and {speaker_b}, session: {session_count}")
    
    qa_pairs = conv_data['qa']
    qa = qa_pairs[0]
    question = qa["question"]
    original_answer = qa.get("answer", "")
    category = qa["category"]

    system_answer = memo.get_response(query=question)
    sample_qa_result = {
        "sample_id": sample_id,
        "qa": {
            "question": question,
            "answer": original_answer,
            "category": category,
            "response": system_answer,
            "retrieved": []
            }
        }
    return sample_qa_result

def split_into_batches(total_samples, batch_size):
    """Split sample indices into batches."""
    batches = []
    for i in range(0, total_samples, batch_size):
        batch = list(range(i, min(i + batch_size, total_samples)))
        batches.append(batch)
    return batches

def process_batch(
    batch_indices,
    batch_id,
    dataset,
    memory_path,
    output_path,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    llm_model=DEFAULT_LLM_MODEL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
):
    batch_results = []
    data_storage_path = os.path.join(memory_path, f"mem_data_batch_{batch_id}")
    batch_output_path = output_path.replace('.json', f'_batch_{batch_id}.json')
    os.makedirs(data_storage_path, exist_ok=True)
    os.makedirs(os.path.dirname(batch_output_path) or ".", exist_ok=True)
    exist_sample = []
    if os.path.exists(batch_output_path):
        with open(batch_output_path, 'r', encoding='utf-8') as f:
            batch_results = json.load(f)
            exist_sample = batch_indices[:len(batch_results)]
            print(f"Batch {batch_id} already processed samples: {exist_sample}, skipping...")
    for i in batch_indices:
        if i in exist_sample:
            continue
        conv_data = dataset[i]
        print(f"Processing sample {i} in batch {batch_id}")
        sample_qa_result = simple_sample(
            conv_data,
            data_storage_path,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
            embedding_model_name=embedding_model_name,
        )
        batch_results.append(sample_qa_result)
        with open(batch_output_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
    return batch_results

def run_memoryos(
    dataset_path,
    output_path,
    memory_path,
    batch_size=25,
    num_processes=None,
    llm_model=DEFAULT_LLM_MODEL,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
):
    llm_model = llm_model or DEFAULT_LLM_MODEL
    llm_api_key = llm_api_key or DEFAULT_LLM_API_KEY
    llm_base_url = llm_base_url or DEFAULT_LLM_BASE_URL
    embedding_model_name = embedding_model_name or DEFAULT_EMBEDDING_MODEL_NAME

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"Using batch processing with batch_size={batch_size}")
    os.makedirs(memory_path, exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Split work into batches.
    total_samples = len(dataset)
    batches = split_into_batches(total_samples, batch_size)
    print(f"Split into {len(batches)} batches: {batches}")

    # Determine the number of processes.
    if num_processes is None:
        num_processes = min(len(batches), mp.cpu_count())
    print(f"Using {num_processes} processes")

    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks.
        future_to_batch = {}
        for batch_id, batch_indices in enumerate(batches):
            future = executor.submit(
                process_batch,
                batch_indices,
                batch_id,
                dataset,
                memory_path,
                output_path,
                llm_api_key,
                llm_base_url,
                llm_model,
                embedding_model_name,
            )
            future_to_batch[future] = batch_id
        
        # Collect results.
        batch_results_list = []
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_id = future_to_batch[future]
            try:
                batch_results = future.result()
                batch_results_list.append((batch_id, batch_results))
                print(f"Batch {batch_id} completed successfully")
            except Exception as exc:
                print(f"Batch {batch_id} generated an exception: {exc}")
                # You can either continue processing other batches or re-raise the exception.
                raise exc
        
        # Sort by batch ID and merge all results.
        batch_results_list.sort(key=lambda x: x[0])
        for batch_id, batch_results in batch_results_list:
            all_results.extend(batch_results)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    return all_results


# if __name__ == "__main__":
#     dataset_path = "./Dataset/LONGMEMEVAL/longmemeval_3.json"
#     data_storage_path = "./Result/LONGMEMEVAL/memoryos/test0903/mem_data"
#     output_path = "./Result/LONGMEMEVAL/memoryos/test0903/results.json"
#     with open(dataset_path, 'r', encoding='utf-8') as f:
#         dataset = json.load(f)
#     all_results = []
#     for conv_data in dataset:
#         sample_qa_result = simple_sample(conv_data, data_storage_path)
#         all_results.append(sample_qa_result)
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(all_results, f, ensure_ascii=False, indent=2)
