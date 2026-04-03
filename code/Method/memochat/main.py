import os
import re
import json
import time
from random import sample

from openai import OpenAI
from transformers.models.gpt2 import GPT2TokenizerFast

# Default configuration
DEFAULT_MODEL_ID = "/home/docker/LLaMA-Factory/output/qwen2_5_lora_sft"
DEFAULT_BASE_URL = "http://localhost:8001/v1"
DEFAULT_API_KEY = "1"
DEFAULT_TOKEN_FILE = "/home/docker/IndepthMem/Result/LOCOMO/memochat/test/token_tracker.json"

q_pre = ""
qa_link = ""
MaxLen = 2048
TarLen = 512
TaskTarLen = {
    "chatting_dialogsum": MaxLen,
    "chatting_alpacagpt4": MaxLen,
    "writing_topiocqa": TarLen // 2,
    "writing_dialogsum": TarLen,
    "retrieval_dialogsum": 32,
    "retrieval_topiocqa": 32
}


def ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def normalize_model_outputs(model_text):
    extracted_elements = [re.sub(r'\s+', ' ', mt.replace('"', '').replace("'", "")) for mt in re.findall(r"'[^']*'|\"[^\"]*\"|\d+", model_text)]
    model_outputs = []
    ti = 0
    while ti + 7 < len(extracted_elements):
        if extracted_elements[ti] == "topic" and extracted_elements[ti + 2] == "summary" and extracted_elements[ti + 4] == "start" and extracted_elements[ti + 6] == "end":
            try:
                model_outputs.append({"topic": extracted_elements[ti + 1], "summary": extracted_elements[ti + 3], "start": int(extracted_elements[ti + 5]), "end": int(extracted_elements[ti + 7])})
            except:
                pass
        ti += 1
    return model_outputs

def normalize_chatting_outputs(model_outputs):
    def white_space_fix(text):
        lines = text.split("\n")
        result = []
        for line in lines:
            result.append(' '.join(line.split()))
        output = '\n'.join(result)
        return output
    return white_space_fix(model_outputs)

def gen_model_output(input_qs, task_type, client, openai_modelid, encoding):
    input_qs_token_l = len(encoding.encode(input_qs))  # token num
    input_qs_word_l = len(input_qs.split(" "))  # word num
    qs_w_t_ratio = input_qs_word_l / input_qs_token_l
    max_word_num = int((MaxLen - TarLen) * qs_w_t_ratio)
    input_qs = " ".join(input_qs.split(" ")[-max_word_num:])
    target_len = TaskTarLen[task_type]
    messages = [{"role": "system", "content": input_qs}]
    # for _ in range(5):
    #     try:
    #         chat = openai.ChatCompletion.create(
    #             model=openai_modelid, messages=messages, max_tokens=target_len, temperature=0.2
    #         )
    #         break
    #     except:
    #         time.sleep(5)
    # model_outputs = chat.choices[0].message.content
    for _ in range(5):
        try:
            chat = client.chat.completions.create(
                model=openai_modelid,
                messages=messages,
                max_tokens=target_len,
                temperature=0.2
            )
            break
        except Exception as e:
            print(f"Model call failed: {e}. Retrying in 5 seconds.")
            time.sleep(5)
    else:
        raise RuntimeError("Model call failed after 5 consecutive retries.")
    model_outputs = chat.choices[0].message.content
    return model_outputs

def run_summary(history, memo, bot_thinking, prompts, client, openai_modelid, encoding):
    system_insturction = prompts["writing_dialogsum"]["system"]
    task_instruction = prompts["writing_dialogsum"]["instruction"]
    # Use the full dialog history here because this pipeline does not include an initial greeting.
    history_log = "\n\n```\nTask Conversation:\n" + "\n".join(["(line {}) {}".format(h_i + 1, h.replace("\n", " ")) for h_i, h in enumerate(history["Recent Dialogs"])])
    qs = q_pre + system_insturction.replace("LINE", str(len(history["Recent Dialogs"]))) + history_log + "\n```" + task_instruction.replace("LINE", str(len(history["Recent Dialogs"]))) + qa_link
    # print("-" * 20 + "summarizing" + "-" * 20)
    # print(qs)
    # print("-" * 20 + "summarizing" + "-" * 20)
    sum_history = gen_model_output(qs, "writing_dialogsum", client, openai_modelid, encoding)
    sum_history = normalize_model_outputs(sum_history)
    # print("-" * 20 + "summarization" + "-" * 20)
    # print(sum_history)
    # print("-" * 20 + "summarization" + "-" * 20)
    for s in sum_history:
        memo[s["topic"]] = memo.get(s["topic"], []) + [{"summary": s["summary"], "dialogs": history["Recent Dialogs"][(s["start"] - 1):s["end"]]}]
    if len(sum_history) == 0:
        if len(history["Recent Dialogs"]) >= 2:
            si_0, si_1 = sample(list(range(len(history["Recent Dialogs"]))), 2)
            memo["NOTO"].append({"summary": "Partial dialogs about: {} or {}.".format(history["Recent Dialogs"][si_0], history["Recent Dialogs"][si_1]), "dialogs": history["Recent Dialogs"]})
        else:
            memo["NOTO"].append({"summary": "Recent dialogs.", "dialogs": history["Recent Dialogs"]})
    history["Recent Dialogs"] = history["Recent Dialogs"][-2:] if len(history["Recent Dialogs"]) >= 2 else []
    bot_thinking["summarization"] = {"input": qs, "output": sum_history}
    return history, memo, bot_thinking

def run_retrieval(history, memo, bot_thinking, prompts, client, openai_modelid, encoding):
    topics = []
    for k, v in memo.items():
        for vv in v:
            topics.append((k, vv["summary"], vv["dialogs"]))
    system_insturction = prompts["retrieval"]["system"]
    task_instruction = prompts["retrieval"]["instruction"]
    task_case = "```\nQuery Sentence:\n" + history["User Input"][6:] + "\nTopic Options:\n" + \
                "\n".join(["({}) {}".format(v_i + 1, v[0] + ". " + v[1]) for v_i, v in enumerate(topics)]) + "\n```"
    qs = q_pre + system_insturction.replace("OPTION", str(len(topics))) + task_case + task_instruction.replace("OPTION", str(len(topics))) + qa_link
    # print("-" * 20 + "retrieving" + "-" * 20)
    # print(qs)
    # print("-" * 20 + "retrieving" + "-" * 20)
    outputs = gen_model_output(qs, "retrieval_dialogsum", client, openai_modelid, encoding)
    # print("-" * 20 + "retrieval" + "-" * 20)
    # print(outputs)
    # print("-" * 20 + "retrieval" + "-" * 20)
    if outputs is None:
        outputs = ""
    outputs = outputs.split("#")
    chosen_topics = []
    for output in outputs:
        try:
            index_ = int(output) - 1
        except:
            continue
        if index_ < len(topics) and "NOTO" not in topics[index_]:
            chosen_topics.append(topics[index_])
    if len(chosen_topics) > 0:
        history["Related Topics"] = [ct[0] for ct in chosen_topics]
        history["Related Summaries"] = [ct[1] for ct in chosen_topics]
        history["Related Dialogs"] = [" ### ".join(ct[2]) for ct in chosen_topics]
    else:
        history["Related Topics"] = []
        history["Related Summaries"] = []
        history["Related Dialogs"] = []
    bot_thinking["retrieval"] = {"input": qs, "output": outputs}
    return history, bot_thinking

def run_memochat(input_data, results_output_path, memory_output_path, prompt_path, 
                 openai_modelid=DEFAULT_MODEL_ID, base_url=DEFAULT_BASE_URL, api_key=DEFAULT_API_KEY, 
                 token_file=DEFAULT_TOKEN_FILE):
    """
    Run the MemoChat conversational memory pipeline.

    Args:
        input_data (str): Path to the input dataset file.
        results_output_path (str): Path to the result output file.
        memory_output_path (str): Path to the memory output file.
        prompt_path (str): Path to the prompt file.
        openai_modelid (str): Model ID. Defaults to DEFAULT_MODEL_ID.
        base_url (str): API base URL. Defaults to DEFAULT_BASE_URL.
        api_key (str): API key. Defaults to DEFAULT_API_KEY.
        token_file (str): Token tracker output path. Reserved for optional tracking.
    """
    print("Starting MemoChat...")
    # tracker = TokenTracker(output_file=token_file)
    # tracker.patch_llm_api()

    openai_modelid = openai_modelid or DEFAULT_MODEL_ID
    base_url = base_url or DEFAULT_BASE_URL
    api_key = api_key or DEFAULT_API_KEY
    token_file = token_file or DEFAULT_TOKEN_FILE

    # Initialize the tokenizer and client.
    encoding = GPT2TokenizerFast.from_pretrained(openai_modelid)
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompts = load_json_file(prompt_path)
    data = load_json_file(input_data)

    ensure_parent_dir(results_output_path)
    ensure_parent_dir(memory_output_path)

    # Resume from existing outputs when available.
    processed_samples = set()
    results = []
    memory = []
    
    try:
        with open(results_output_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
            for item in existing_results:
                processed_samples.add(item["sample_id"])
                results.append(item)
        print(f"Loaded {len(processed_samples)} processed samples from the existing results file.")
    except FileNotFoundError:
        print("No existing results file found. Starting from scratch.")
    
    try:
        with open(memory_output_path, "r", encoding="utf-8") as f:
            memory = json.load(f)
    except FileNotFoundError:
        memory = []
    
    count = 0
    for d in data:
        sample_id = d["sample_id"]
        
        # Skip samples that have already been processed.
        if sample_id in processed_samples:
            print(f"Skipping processed sample: {sample_id}")
            continue
            
        count += 1
        print("=" * 20 + "start of conversation {}".format(sample_id) + "=" * 20)
        
        sample_result = {
            "sample_id": sample_id,
            "qa": []
        }

        history = {
            "Recent Dialogs": [], 
            "Related Topics": [], 
            "Related Summaries": [], 
            "Related Dialogs": [], 
            "User Input": "",
        }
        memo = {
            "NOTO": [{"summary": "None of the others.", "dialogs": []}]
        }

        # Process all dialog sessions.
        conversation_data = d["conversation"]
        
        # Collect all session keys dynamically and process them in order.
        session_keys = [key for key in conversation_data.keys() if key.startswith("session_") and not key.endswith("_date_time")]
        session_keys.sort()  # Keep the order as session_1, session_2, session_3, ...
        
        # with tracker.stage(f"Sample {sample_id}"):
        for session_key in session_keys:
            print("Processing {}...".format(session_key))
            session_dialogs = conversation_data[session_key]
            
            # Read the session timestamp if it is available.
            session_time_key = session_key + "_date_time"
            session_time = conversation_data.get(session_time_key, "Unknown time")
            print("Session time: {}".format(session_time))
            
            # with tracker.stage(f"Session {session_key}"):
            dial_id = 0
            for dialog in session_dialogs:
                # with tracker.stage(f"Dialog {dial_id}"):
                speaker = dialog["speaker"]
                text = dialog["text"]
                # Keep timestamp and dialog ID inside the stored dialog line.
                dia_id = dialog.get("dia_id", "")
                dialog_line = "{}: {} [Time: {}, ID: {}]".format(speaker, text, session_time, dia_id)
                history["Recent Dialogs"].append(dialog_line)
                # print("Added dialog: {}".format(dialog_line))

                # Summarize when the recent window grows too large.
                if len(" ### ".join(history["Recent Dialogs"]).split(" ")) > (MaxLen // 2) or len(history["Recent Dialogs"]) >= 10:
                    print("Summarization threshold reached. Starting summarization...")
                    bot_thinking = {"retrieval": "", "summarization": ""}
                    history, memo, bot_thinking = run_summary(history, memo, bot_thinking, prompts, client, openai_modelid, encoding)
                    print("Summarization finished. Current memo topics: {}".format(list(memo.keys())))
                dial_id += 1

        # Run a final summarization pass after all dialogs are processed.
        if len(history["Recent Dialogs"]) > 0:
            print("Conversation finished. Running final summarization...")
            bot_thinking = {"retrieval": "", "summarization": ""}
            history, memo, bot_thinking = run_summary(history, memo, bot_thinking, prompts, client, openai_modelid, encoding)
            print("Final summarization finished.")

        # Answer QA items.
        if "qa" in d:
            print("Starting QA answering...")
            # with tracker.stage(f"Sample {sample_id}"):
            qa_idx = 0
            for qa_item in d["qa"]:
                # with tracker.stage(f"Processing QA {qa_idx}"):
                question = qa_item["question"]
                print("Question: {}".format(question))
                
                # Retrieve the most relevant memory items for the current question.
                history["User Input"] = "user: " + question
                if len(memo.keys()) > 1:
                    history, bot_thinking = run_retrieval(history, memo, bot_thinking, prompts, client, openai_modelid, encoding)
                    
                    # Generate the answer using the retrieved evidence.
                    system_instruction = prompts["chatting"]["system"]
                    task_instruction = prompts["chatting"]["instruction"]
                    
                    # Provide temporal context for every question.
                    task_case = "```\nRelated Evidences (with temporal information):\n" + "\n".join(["({}) {}".format(r_tsd_i + 1, {
                                    "Related Topics": history["Related Topics"][r_tsd_i], 
                                    "Related Summaries": history["Related Summaries"][r_tsd_i], 
                                    "Related Dialogs": history["Related Dialogs"][r_tsd_i]
                                }) for r_tsd_i in range(len(history["Related Topics"]))]) + "\n\nQuestion:\n" + question + "\n\nNote: Pay special attention to timestamps [Time: ...] and dialog IDs [ID: ...] in the dialogs to provide accurate answers.\n```"
                    
                    qs = q_pre + system_instruction + task_case + task_instruction + qa_link + "\n# Note:\nThe answer must be brief (under 5-6 words) and direct, with no extra description."
                    answer = gen_model_output(qs, "chatting_dialogsum", client, openai_modelid, encoding)
                    answer = normalize_chatting_outputs(answer)
                else:
                    answer = "No relevant information found in the dialog memory."
                
                # Preserve the original QA category when it exists.
                category = qa_item.get("category", 1)
                
                qa_result = {
                    "question": question,
                    "answer": qa_item.get("answer"),  # Keep the reference answer when it exists.
                    "category": category,
                    "response": answer,
                    "retrieved": history["Related Summaries"] + history["Related Dialogs"]
                }
                sample_result["qa"].append(qa_result)
                print("Answer: {}".format(answer))
                print("-" * 50)
                qa_idx += 1

        # Save progress after each sample.
        results.append(sample_result)
        memory.append(memo)
        
        # Create parent folders before each incremental save.
        ensure_parent_dir(results_output_path)
        ensure_parent_dir(memory_output_path)
        with open(results_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(memory_output_path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
        
        print("=" * 20 + "end of conversation {}".format(sample_id) + "=" * 20)
        print("Final memo topics: {}".format(list(memo.keys())))
        print("QA count: {}".format(len(sample_result["qa"])))
        print("Saved results to disk.")
        print("\n")

    # Final status message.
    print(f"Processing finished. Total samples saved: {len(results)}")
    return results, memory

if __name__ == "__main__":
    # Default paths for direct execution.
    default_input_data = "/home/docker/IndepthMem/Dataset/LOCOMO/locomodemo.json"
    default_results_output_path = "/home/docker/IndepthMem/Result/LOCOMO/memochat/result.json"
    default_memory_output_path = "/home/docker/IndepthMem/Result/LOCOMO/memochat/memory.json"
    default_prompt_path = "/home/docker/IndepthMem/Method/memochat/prompt_loco.json"
    
    # Run with the default configuration.
    results, memory = run_memochat(
        input_data=default_input_data,
        results_output_path=default_results_output_path,
        memory_output_path=default_memory_output_path,
        prompt_path=default_prompt_path
    )
