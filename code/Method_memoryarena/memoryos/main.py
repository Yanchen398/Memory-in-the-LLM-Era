import json
import contextlib
from datetime import datetime, timedelta
from .short_term_memory import ShortTermMemory
from .mid_term_memory import MidTermMemory
from .long_term_memory import LongTermMemory
from .dynamic_update import DynamicUpdate
from .retrieval_and_answer import RetrievalAndAnswer
from .utils import DEFAULT_EMBEDDING_MODEL_NAME, DEFAULT_LLM_API_KEY, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, build_default_client, configure_memoryos_runtime, gpt_generate_answer, gpt_extract_theme, gpt_update_profile, gpt_generate_multi_summary, get_timestamp, llm_extract_keywords, gpt_personality_analysis
import re
import openai
import time
import tiktoken
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from .token_tracker import TokenTracker
from ..dataset_hygiene import natural_session_keys, pair_session_turns, raw_result_path

total_tokens = 0
num_samples=0

# Heat threshold
H_THRESHOLD = 5.0

def update_user_profile_from_top_segment(mid_mem, long_mem, sample_id, client):
    """
    Update user profile if heat exceeds threshold and extract assistant knowledge.
    """
    if not mid_mem.heap:
        return
    
    neg_heat, sid = mid_mem.heap[0]
    mid_mem.rebuild_heap()
    current_heat = -neg_heat
    
    if current_heat >= H_THRESHOLD:
        session = mid_mem.sessions.get(sid)
        if not session:
            return
        
        un_analyzed = [p for p in session["details"] if not p.get("analyzed", False)]
        if un_analyzed:
            print(f"Updating user profile: Segment {sid} heat {current_heat:.2f} exceeds threshold, starting profile update...")
            
            old_profile = long_mem.get_raw_user_profile(sample_id)
            
            result = gpt_personality_analysis(un_analyzed, client)
            new_profile = result["profile"]
            new_private = result["private"]
            assistant_knowledge = result["assistant_knowledge"]
            
            if old_profile:
                updated_profile = gpt_update_profile(old_profile, new_profile, client)
            else:
                updated_profile = new_profile
                
            long_mem.update_user_profile(sample_id, updated_profile)
            
            # Split new_private into individual facts and store them one by one.
            if new_private and new_private != "- None":
                # Split by line and ignore empty or non-factual lines such as "【User Data】" or comments.
                facts = [line.strip() for line in new_private.split("\n")]
                for fact in facts:
                    long_mem.add_knowledge(fact)  # Add each fact individually.
            
            if assistant_knowledge and assistant_knowledge != "None":
                long_mem.add_assistant_knowledge(assistant_knowledge)
            
            for p in session["details"]:
                p["analyzed"] = True
            session["N_visit"] = 0
            session["L_interaction"] = 0
            session["R_recency"] = 1.0
            session["H_segment"] = 0.0
            session["last_visit_time"] = get_timestamp()
            mid_mem.rebuild_heap()
            mid_mem.save()
            print(f"Update complete: Segment {sid} heat has been reset.")

def generate_system_response_with_meta(query, short_mem, long_mem, retrieval_queue, long_konwledge, client, llm_model, sample_id, speaker_a, speaker_b, meta_data):
    """
    Generate system response with speaker roles clearly defined.
    """
    history = short_mem.get_all()
    history_text = "\n".join([
        f"{speaker_a}: {qa.get('user_input', '')}\n{speaker_b}: {qa.get('agent_response', '')}\nTime: ({qa.get('timestamp', '')})" 
        for qa in history
    ])
    
    retrieval_text = "\n".join([
        f"【Historical Memory】 {speaker_a}: {page.get('user_input', '')}\n{speaker_b}: {page.get('agent_response', '')}\nTime:({page.get('timestamp', '')})\nConversation chain overview:({page.get('meta_info', '')})\n" 
        for page in retrieval_queue
    ])
    
    profile_obj = long_mem.get_user_profile(sample_id)
    user_profile_text = str(profile_obj.get("data", "None")) if profile_obj else "None"
    
    background = f"【User Profile】\n{user_profile_text}\n\n"
    for kn in long_konwledge:
        background += f"{kn['knowledge']}\n"
    background = re.sub(r'(?i)\buser\b', speaker_a, background)
    background= re.sub(r'(?i)\bassistant\b', speaker_b, background)
    assistant_knowledge = long_mem.get_assistant_knowledge()
    assistant_knowledge_text = "【Assistant Knowledge】\n"
    for ak in assistant_knowledge:
        assistant_knowledge_text += f"- {ak['knowledge']} ({ak['timestamp']})\n"
    #meta_data_text = f"【Conversation Meta Data】\n{json.dumps(meta_data, ensure_ascii=False, indent=2)}\n\n"
    assistant_knowledge_text = re.sub(r'\bI\b', speaker_b, assistant_knowledge_text)
    
    system_prompt = (
        f"You are role-playing as {speaker_b} in a conversation with the user is playing as {speaker_a}. "
        f"Here are some of your character traits and knowledge:\n{assistant_knowledge_text}\n"
        f"Any content referring to 'User' in the prompt refers to {speaker_a}'s content, and any content referring to 'AI'or 'assiant' refers to {speaker_b}'s content."
        f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
        f"When the question is: \"What did the charity race raise awareness for?\", you should not answer in the form of: \"The charity race raised awareness for mental health.\" Instead, it should be: \"mental health\", as this is more concise."
    )

    user_prompt = (
        f"<CONTEXT>\n"
        f"Recent conversation between {speaker_a} and {speaker_b}:\n"
        f"{history_text}\n\n"
        f"<MEMORY>\n"
        f"Relevant past conversations:\n"
        f"{retrieval_text}\n\n"
        f"<CHARACTER TRAITS>\n"
        f"Characteristics of {speaker_a}:\n"
        f"{background}\n\n"
        f"the question is: {query}\n"
        f"Your task is to answer questions about {speaker_a} or {speaker_b} in an extremely concise manner.\n"
        f"Please only provide the content of the answer, without including 'answer:'\n"
        f"For questions that require answering a date or time, strictly follow the format \"15 July 2023\" and provide a specific date whenever possible. For example, if you need to answer \"last year,\" give the specific year of last year rather than just saying \"last year.\" Only provide one year, date, or time, without any extra responses.\n"
        f"If the question is about the duration, answer in the form of several years, months, or days.\n"
        f"Generate answers primarily composed of concrete entities, such as Mentoring program, school speech, etc"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat_completion(model=llm_model, messages=messages, temperature=0.0, max_tokens=2000)
    return response, system_prompt, user_prompt

def process_conversation(conversation_data):
    """
    Process conversation data from locomo10 format into memory system format.
    Handles both text-only and image-containing messages.
    """
    processed = []
    speaker_a = conversation_data["speaker_a"]
    speaker_b = conversation_data["speaker_b"]
    
    session_keys = natural_session_keys(conversation_data)
    
    for session_key in session_keys:
        timestamp_key = f"{session_key}_date_time"
        timestamp = conversation_data.get(timestamp_key, "")
        
        for exchange in pair_session_turns(
            conversation_data, session_key, speaker_a, speaker_b
        ):
            query = exchange["query"]
            response = exchange["response"]
            processed.append(
                {
                    "user_input": query.split(": ", 1)[1] if query.startswith(f"{speaker_a}: ") else query,
                    "agent_response": (
                        response.split(": ", 1)[1]
                        if response.startswith(f"{speaker_b}: ")
                        else response
                    ),
                    "timestamp": timestamp,
                    "source_turn_indices": exchange["source_turn_indices"],
                }
            )
    
    return processed


def parse_retrieve_top_ks(retrieve_top_ks):
    if retrieve_top_ks is None:
        return [10]
    if isinstance(retrieve_top_ks, int):
        return [retrieve_top_ks]
    if isinstance(retrieve_top_ks, str):
        values = [part.strip() for part in retrieve_top_ks.split(",") if part.strip()]
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
    if len(retrieve_top_ks) == 1 and output_path.endswith(".json"):
        return {retrieve_top_ks[0]: raw_result_path(output_path)}
    output_root = output_path if not output_path.endswith(".json") else os.path.dirname(output_path)
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
                with open(output_file, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                if isinstance(existing_results, list):
                    results = existing_results
                    processed = {result.get("sample_id") for result in results if result.get("sample_id")}
                    print(f"Existing top_k={k} results detected: {len(processed)} processed samples.")
            except Exception as e:
                print(f"Error while reading {output_file}: {e}. Restarting that top-k from scratch.")
        else:
            print(f"No existing results file for top_k={k}. Starting from scratch.")
        results_by_k[k] = results
        processed_by_k[k] = processed
    return results_by_k, processed_by_k


def save_results(output_file, results):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def reset_sample_memory_files(memory_path, sample_id):
    for suffix in ("short_term", "mid_term", "long_term"):
        path = os.path.join(memory_path, f"{sample_id}_{suffix}.json")
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed partial memory file before rebuilding sample: {path}")


class NullTracker:
    def stage(self, name):
        return contextlib.nullcontext()


def run_memoryos(
    dataset_path,
    output_path,
    memory_path,
    llm_model=DEFAULT_LLM_MODEL,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_api_key="EMPTY",
    embedding_base_url=None,
    token_file=None,
    retrieve_top_ks=None,
    short_term_capacity=1,
    qa_concurrency=1,
    track_tokens=True,
    fast_index=False,
    update_profiles=True,
    use_retrieval_keywords=True,
    memory_granularity="message",
    segment_threshold=0.5,
    segment_max_messages=0,
):
    llm_model = llm_model or DEFAULT_LLM_MODEL
    llm_api_key = llm_api_key or DEFAULT_LLM_API_KEY
    llm_base_url = llm_base_url or DEFAULT_LLM_BASE_URL
    embedding_model_name = embedding_model_name or DEFAULT_EMBEDDING_MODEL_NAME
    configure_memoryos_runtime(
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        embedding_model_name=embedding_model_name,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
    )
    client = build_default_client()
    if not token_file:
        output_root = output_path if not output_path.endswith(".json") else os.path.dirname(output_path)
        token_file = os.path.join(os.path.abspath(output_root or "."), "token_tracker.json")
    if track_tokens:
        tracker = TokenTracker(output_file=token_file)
        tracker.patch_llm_api()
    else:
        tracker = NullTracker()
        print("Token tracking disabled for this run.")
    short_term_capacity = int(short_term_capacity or 1)
    qa_concurrency = max(1, int(qa_concurrency or 1))
    fast_index = bool(fast_index)
    update_profiles = bool(update_profiles)
    use_retrieval_keywords = bool(use_retrieval_keywords)
    memory_granularity = str(memory_granularity or "message").strip().lower()
    if memory_granularity not in {"message", "segment"}:
        raise ValueError("memory_granularity must be 'message' or 'segment'")

    dynamic_update_class = DynamicUpdate
    dynamic_update_kwargs = {}
    if memory_granularity == "segment":
        from Method.sota_abl.memoryos_segment_update import SegmentDynamicUpdate

        dynamic_update_class = SegmentDynamicUpdate
        dynamic_update_kwargs = {
            "segment_threshold": segment_threshold,
            "segment_max_messages": segment_max_messages,
        }

    print(
        "Starting processing for the full locomo10 dataset "
        f"(memory_granularity={memory_granularity})..."
    )
    
    # Create the memory storage directory.
    os.makedirs(memory_path, exist_ok=True)
    
    # Load locomo10 dataset
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"Dataset loaded successfully with {len(dataset)} samples.")
    except FileNotFoundError:
        print("Error: locomo10.json could not be found. Please make sure the file exists.")
        return
    except Exception as e:
        print(f"Error while loading the dataset: {e}")
        return
    
    # Process the full dataset without slicing.
    # dataset = dataset  # Process the entire dataset.
    
    retrieve_top_ks = parse_retrieve_top_ks(retrieve_top_ks)
    output_paths = build_topk_output_paths(output_path, retrieve_top_ks)
    for top_k, output_file in output_paths.items():
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        print(f"top_k={top_k} results will be saved to {output_file}")

    results_by_k, processed_by_k = load_existing_results_by_k(output_paths)

    total_samples = len(dataset)
    
    for idx, sample in enumerate(dataset):
        sample_id = sample.get("sample_id", "unknown_sample")
        
        # Skip only when every requested top-k result already contains this sample.
        if all(sample_id in processed_by_k[k] for k in retrieve_top_ks):
            print(f"Sample {idx + 1}/{total_samples}: {sample_id} has already been processed for all top-k values. Skipping.")
            continue
            
        print(f"Processing sample {idx + 1}/{total_samples}: {sample_id}")
        
        conversation_data = sample["conversation"]
        qa_pairs = sample["qa"]
        
        # Process conversation data
        processed_dialogs = process_conversation(conversation_data)
        
        if not processed_dialogs:
            print(f"Sample {sample_id} has no valid dialog data. Skipping.")
            continue
            
        speaker_a = conversation_data["speaker_a"]
        speaker_b = conversation_data["speaker_b"]

        # If a previous run failed inside this sample, its memory files may exist
        # even though no result was saved. Rebuild this sample cleanly on resume.
        reset_sample_memory_files(memory_path, sample_id)
        
        # Initialize memory modules
        short_mem = ShortTermMemory(max_capacity=short_term_capacity, file_path=os.path.join(memory_path, f"{sample_id}_short_term.json"))
        mid_mem = MidTermMemory(max_capacity=2000, file_path=os.path.join(memory_path, f"{sample_id}_mid_term.json"), client=client)
        long_mem = LongTermMemory(file_path=os.path.join(memory_path, f"{sample_id}_long_term.json"))
        dynamic_updater = dynamic_update_class(
            short_mem,
            mid_mem,
            long_mem,
            topic_similarity_threshold=0.6,
            client=client,
            llm_model=llm_model,
            fast_index=fast_index,
            **dynamic_update_kwargs,
        )
        retrieval_system = RetrievalAndAnswer(short_mem, mid_mem, long_mem, dynamic_updater, queue_capacity=max(retrieve_top_ks))
        
        # Store conversation history in memory system
        with tracker.stage(f"Sample {sample_id}"):
            dial_id = 0
            for dialog in processed_dialogs:
                print(f"Processing {dial_id}:{dialog}")
                with tracker.stage(f"Dialog {dial_id}"):
                    short_mem.add_qa_pair(dialog)
                    if short_mem.is_full():
                        dynamic_updater.bulk_evict_and_update_mid_term()
                    if update_profiles:
                        update_user_profile_from_top_segment(mid_mem, long_mem, sample_id, client)
                dial_id += 1
            if short_mem.get_all():
                print("Flushing remaining short-term memory into the index...")
                dynamic_updater.bulk_evict_and_update_mid_term(force=True)
                if update_profiles:
                    update_user_profile_from_top_segment(mid_mem, long_mem, sample_id, client)
        
        # Process QA pairs for current sample. Each query is retrieved once at max(top_k),
        # then the ranked list is truncated for each requested top-k result.
        sample_qa_results_by_k = {k: [] for k in retrieve_top_ks}
        qa_count = len(qa_pairs)
        max_retrieve_top_k = max(retrieve_top_ks)

        def process_qa_item(qa_idx, qa):
            print(f"  Processing QA {qa_idx + 1}/{qa_count}")
            question = qa["question"]
            original_answer = qa.get("answer", "")
            category = qa["category"]
            evidence = qa.get("evidence", "")
            if(original_answer == ""):
                original_answer = qa.get("adversarial_answer", "")

            meta_data = {
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "category": category,
                "evidence": evidence
            }

            local_retrieval_system = RetrievalAndAnswer(short_mem, mid_mem, long_mem, dynamic_updater, queue_capacity=max_retrieve_top_k)
            retrieval_start = time.perf_counter()
            retrieval_result = local_retrieval_system.retrieve(
                question,
                segment_threshold=0.1,
                page_threshold=0.1,
                knowledge_threshold=0.1,
                client=client,
                top_k=max_retrieve_top_k,
                update_stats=False,
                use_llm_keywords=use_retrieval_keywords,
            )
            retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

            per_k_results = {}
            for retrieve_top_k in retrieve_top_ks:
                if sample_id in processed_by_k[retrieve_top_k]:
                    continue
                top_queue = retrieval_result["retrieval_queue"][:retrieve_top_k]
                top_knowledge = retrieval_result["long_term_knowledge"][:retrieve_top_k]
                system_answer, system_prompt, user_prompt = generate_system_response_with_meta(
                    question,
                    short_mem,
                    long_mem,
                    top_queue,
                    top_knowledge,
                    client,
                    llm_model,
                    sample_id,
                    speaker_a,
                    speaker_b,
                    meta_data
                )

                retrieved = []
                for item in top_queue:
                    if item.get("user_input"):
                        retrieved.append(item["user_input"])
                    if item.get("agent_response"):
                        retrieved.append(item["agent_response"])
                for knowledge_item in top_knowledge:
                    retrieved.append(knowledge_item['knowledge'])

                per_k_results[retrieve_top_k] = {
                    "question": question,
                    "answer": original_answer,
                    "category": category,
                    "response": system_answer,
                    "retrieved": retrieved,
                    "retrieval_top_k": retrieve_top_k,
                    "retrieval_latency_ms": retrieval_latency_ms
                }
            return qa_idx, per_k_results

        with tracker.stage(f"Sample {sample_id}"):
            if qa_concurrency == 1:
                for qa_idx, qa in enumerate(qa_pairs):
                    _, per_k_results = process_qa_item(qa_idx, qa)
                    for retrieve_top_k, qa_result in per_k_results.items():
                        sample_qa_results_by_k[retrieve_top_k].append(qa_result)
            else:
                ordered_results = {k: [None] * qa_count for k in retrieve_top_ks}
                with ThreadPoolExecutor(max_workers=qa_concurrency) as executor:
                    futures = [executor.submit(process_qa_item, qa_idx, qa) for qa_idx, qa in enumerate(qa_pairs)]
                    for future in as_completed(futures):
                        qa_idx, per_k_results = future.result()
                        for retrieve_top_k, qa_result in per_k_results.items():
                            ordered_results[retrieve_top_k][qa_idx] = qa_result
                for retrieve_top_k in retrieve_top_ks:
                    sample_qa_results_by_k[retrieve_top_k] = [item for item in ordered_results[retrieve_top_k] if item is not None]

        # Save results after each sample for real-time progress persistence.
        for retrieve_top_k in retrieve_top_ks:
            if sample_id in processed_by_k[retrieve_top_k]:
                continue
            results_by_k[retrieve_top_k].append({
                "sample_id": sample_id,
                "qa": sample_qa_results_by_k[retrieve_top_k]
            })
            processed_by_k[retrieve_top_k].add(sample_id)
            try:
                save_results(output_paths[retrieve_top_k], results_by_k[retrieve_top_k])
                print(f"Sample {idx + 1} completed for top_k={retrieve_top_k}. Results saved to {output_paths[retrieve_top_k]}")
            except Exception as e:
                print(f"Error while saving top_k={retrieve_top_k} results: {e}")

    # Final save and latency manifest.
    manifest = {"retrieve_top_ks": retrieve_top_ks, "outputs": {}}
    for retrieve_top_k in retrieve_top_ks:
        try:
            save_results(output_paths[retrieve_top_k], results_by_k[retrieve_top_k])
        except Exception as e:
            print(f"Error during final top_k={retrieve_top_k} result save: {e}")
        latencies = [
            qa.get("retrieval_latency_ms")
            for sample in results_by_k[retrieve_top_k]
            for qa in sample.get("qa", [])
            if isinstance(qa.get("retrieval_latency_ms"), (int, float))
        ]
        manifest["outputs"][str(retrieve_top_k)] = {
            "result_file": output_paths[retrieve_top_k],
            "sample_count": len(results_by_k[retrieve_top_k]),
            "qa_count": sum(len(sample.get("qa", [])) for sample in results_by_k[retrieve_top_k]),
            "average_retrieval_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        }
    manifest_path = os.path.join(output_path if not output_path.endswith(".json") else os.path.dirname(output_path), "memoryos_run_manifest.json")
    save_results(manifest_path, manifest)

    return results_by_k

if __name__ == "__main__":
    run_memoryos(dataset_path="./Dataset/LOCOMO/locomodemo.json", output_path="./Result/LOCOMO/memoryos/demo.json", memory_path="./Result/LOCOMO/memoryos/mem_tmp_loco_final")
