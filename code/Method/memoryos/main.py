import json
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

from .token_tracker import TokenTracker

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
    
    response = client.chat_completion(model=llm_model, messages=messages, temperature=0.7, max_tokens=2000)
    return response, system_prompt, user_prompt

def process_conversation(conversation_data):
    """
    Process conversation data from locomo10 format into memory system format.
    Handles both text-only and image-containing messages.
    """
    processed = []
    speaker_a = conversation_data["speaker_a"]
    speaker_b = conversation_data["speaker_b"]
    
    # Find all session keys
    session_keys = [key for key in conversation_data.keys() if key.startswith("session_") and not key.endswith("_date_time")]
    
    for session_key in session_keys:
        timestamp_key = f"{session_key}_date_time"
        timestamp = conversation_data.get(timestamp_key, "")
        
        for dialog in conversation_data[session_key]:
            speaker = dialog["speaker"]
            text = dialog["text"]
            
            # Handle image content if present
            if "blip_caption" in dialog and dialog["blip_caption"]:
                text = f"{text} (image description: {dialog['blip_caption']})"
            
            # Alternate between speakers as user and assistant
            if speaker == speaker_a:
                processed.append({
                    "user_input": text,
                    "agent_response": "",
                    "timestamp": timestamp
                })
            else:
                if processed:
                    processed[-1]["agent_response"] = text
                else:
                    processed.append({
                        "user_input": "",
                        "agent_response": text,
                        "timestamp": timestamp
                    })
    
    return processed

def run_memoryos(
    dataset_path,
    output_path,
    memory_path,
    llm_model=DEFAULT_LLM_MODEL,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
    token_file="/home/docker/IndepthMem/Result/LOCOMO/memoryos/token_tracker.json",
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
    )
    client = build_default_client()
    tracker = TokenTracker(output_file=token_file)
    tracker.patch_llm_api()

    print("Starting processing for the full locomo10 dataset...")
    
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
    
    # Use the provided output file path.
    output_file = output_path
    
    # Resume processing if an output file already exists.
    results = []
    processed_samples = set()
    start_idx = 0
    
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            if existing_results:
                results = existing_results
                processed_samples = {result["sample_id"] for result in results}
                start_idx = len(results)
                print(f"Existing results detected. {len(processed_samples)} samples have already been processed. Resuming from sample {start_idx + 1}.")
        except Exception as e:
            print(f"Error while reading the existing results file: {e}. Restarting from scratch.")
            results = []
            processed_samples = set()
            start_idx = 0
    else:
        print("No existing results file found. Starting from scratch.")
    
    total_samples = len(dataset)
    
    for idx, sample in enumerate(dataset):
        sample_id = sample.get("sample_id", "unknown_sample")
        
        # Skip samples that have already been processed.
        if sample_id in processed_samples:
            print(f"Sample {idx + 1}/{total_samples}: {sample_id} has already been processed. Skipping.")
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
        
        # Initialize memory modules
        short_mem = ShortTermMemory(max_capacity=1, file_path=os.path.join(memory_path, f"{sample_id}_short_term.json"))
        mid_mem = MidTermMemory(max_capacity=2000, file_path=os.path.join(memory_path, f"{sample_id}_mid_term.json"), client=client)
        long_mem = LongTermMemory(file_path=os.path.join(memory_path, f"{sample_id}_long_term.json"))
        dynamic_updater = DynamicUpdate(short_mem, mid_mem, long_mem, topic_similarity_threshold=0.6, client=client, llm_model=llm_model)
        retrieval_system = RetrievalAndAnswer(short_mem, mid_mem, long_mem, dynamic_updater, queue_capacity=10)
        
        # Store conversation history in memory system
        with tracker.stage(f"Sample {sample_id}"):
            dial_id = 0
            for dialog in processed_dialogs:
                print(f"Processing {dial_id}:{dialog}")
                with tracker.stage(f"Dialog {dial_id}"):
                    short_mem.add_qa_pair(dialog)
                    if short_mem.is_full():
                        dynamic_updater.bulk_evict_and_update_mid_term()
                    update_user_profile_from_top_segment(mid_mem, long_mem, sample_id, client)
                dial_id += 1
        
        # Process QA pairs for current sample
        sample_qa_results = []
        qa_count = len(qa_pairs)
        with tracker.stage(f"Sample {sample_id}"):
            for qa_idx, qa in enumerate(qa_pairs):
                with tracker.stage(f"Processing QA {qa_idx}"):
                    print(f"  Processing QA {qa_idx + 1}/{qa_count}")
                    question = qa["question"]
                    original_answer = qa.get("answer", "")
                    category = qa["category"]
                    evidence = qa.get("evidence", "")
                    if(original_answer == ""):
                        original_answer = qa.get("adversarial_answer", "")
                    # Retrieve and generate answer
                    retrieval_result = retrieval_system.retrieve(
                        question, 
                        segment_threshold=0.1, 
                        page_threshold=0.1, 
                        knowledge_threshold=0.1, 
                        client=client
                    )
                    
                    # Generate meta data for the conversation
                    meta_data = {
                        "sample_id": sample_id,
                        "speaker_a": speaker_a,
                        "speaker_b": speaker_b,
                        "category": category,
                        "evidence": evidence
                    }
                    
                    system_answer, system_prompt, user_prompt = generate_system_response_with_meta(
                        question, 
                        short_mem, 
                        long_mem, 
                        retrieval_result["retrieval_queue"], 
                        retrieval_result["long_term_knowledge"],
                        client, 
                        llm_model,
                        sample_id, 
                        speaker_a, 
                        speaker_b, 
                        meta_data
                    )
                    
                    # Build retrieved list from retrieval_queue and long_term_knowledge
                    retrieved = []
                    
                    # Add retrieval_queue items (alternating user_input and agent_response)
                    for item in retrieval_result["retrieval_queue"]:
                        if item.get("user_input"):
                            retrieved.append(item["user_input"])
                        if item.get("agent_response"):
                            retrieved.append(item["agent_response"])
                    
                    # Add long_term_knowledge items
                    for knowledge_item in retrieval_result["long_term_knowledge"]:
                        retrieved.append(knowledge_item['knowledge'])
                    
                    # Save result for the current QA pair in the new format
                    sample_qa_results.append({
                        "question": question,
                        "answer": original_answer,
                        "category": category,
                        "response": system_answer,
                        "retrieved": retrieved
                    })
            
        # Add sample result to overall results
        results.append({
            "sample_id": sample_id,
            "qa": sample_qa_results
        })
    
        # Save results after each sample for real-time progress persistence.
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Sample {idx + 1} completed. Results saved to {output_file}")
        except Exception as e:
            print(f"Error while saving results: {e}")
    
    # Final save.
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error during final result save: {e}")

    return results

if __name__ == "__main__":
    run_memoryos(dataset_path="/home/docker/IndepthMem/Dataset/LOCOMO/locomodemo.json", output_path="/home/docker/IndepthMem/Result/LOCOMO/memoryos/demo.json", memory_path="/home/docker/IndepthMem/Result/LOCOMO/memoryos/mem_tmp_loco_final")
