import argparse
import os
import re
import json
import time
import copy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from random import sample

from openai import OpenAI
from transformers.models.gpt2 import GPT2TokenizerFast

from ..dataset_hygiene import format_turn_text, natural_session_keys, raw_result_path

# Default configuration
DEFAULT_MODEL_ID = "Qwen3.5-9B"
DEFAULT_BASE_URL = "http://localhost:8001/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_TOKEN_FILE = None

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


def write_json_atomic(file_path, payload):
    """Write JSON without exposing a partially-written checkpoint."""
    ensure_parent_dir(file_path)
    temp_path = "{}.tmp.{}".format(file_path, os.getpid())
    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        os.replace(temp_path, file_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


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

def count_tokens(encoding, text):
    if encoding is None:
        return max(1, len(text.split()))
    try:
        return max(1, len(encoding.encode(text)))
    except Exception:
        return max(1, len(text.split()))


def build_tokenizer(openai_modelid):
    for candidate in [openai_modelid, "gpt2"]:
        if not candidate:
            continue
        try:
            return GPT2TokenizerFast.from_pretrained(candidate, local_files_only=True)
        except Exception as exc:
            print(f"Tokenizer load failed for {candidate}: {exc}")
    print("Falling back to whitespace token counting.")
    return None


def gen_model_output(input_qs, task_type, client, openai_modelid, encoding):
    input_qs_token_l = count_tokens(encoding, input_qs)
    input_qs_word_l = max(1, len(input_qs.split(" ")))
    qs_w_t_ratio = input_qs_word_l / input_qs_token_l
    max_word_num = max(1, int((MaxLen - TarLen) * qs_w_t_ratio))
    input_qs = " ".join(input_qs.split(" ")[-max_word_num:])
    target_len = TaskTarLen[task_type]
    messages = [{"role": "user", "content": input_qs}]
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
            try:
                chat = client.chat.completions.create(
                    model=openai_modelid,
                    messages=messages,
                    max_tokens=target_len,
            temperature=0.0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except TypeError:
                chat = client.chat.completions.create(
                    model=openai_modelid,
                    messages=messages,
                    max_tokens=target_len,
                temperature=0.0,
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

def parse_top_ks(value):
    if value is None or value == "":
        return [10]
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    return [int(part) for part in parts]


def resolve_output_files(results_output_path, memory_output_path, top_ks):
    results_output_path = raw_result_path(
        results_output_path or "./Result/LOCOMO/memochat/result_raw.json"
    )
    _, ext = os.path.splitext(results_output_path)
    result_files = {}
    if ext.lower() == ".json":
        base_dir = os.path.dirname(results_output_path) or "."
        base_name = os.path.basename(results_output_path)
        if len(top_ks) == 1:
            result_files[top_ks[0]] = results_output_path
        else:
            for top_k in top_ks:
                result_files[top_k] = os.path.join(base_dir, f"top_k_{top_k}", base_name)
    else:
        base_dir = results_output_path
        for top_k in top_ks:
            result_files[top_k] = raw_result_path(os.path.join(base_dir, f"top_k_{top_k}"))

    if memory_output_path:
        _, memory_ext = os.path.splitext(memory_output_path)
        memory_file = memory_output_path if memory_ext.lower() == ".json" else os.path.join(memory_output_path, "memory.json")
    else:
        memory_file = os.path.join(os.path.dirname(next(iter(result_files.values()))), "memory.json")
    return result_files, memory_file


def load_existing_results(result_files):
    results_by_k = {top_k: [] for top_k in result_files}
    processed_by_k = {top_k: set() for top_k in result_files}
    for top_k, result_file in result_files.items():
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            if isinstance(existing_results, list):
                results_by_k[top_k] = existing_results
                processed_by_k[top_k] = {item.get("sample_id") for item in existing_results if item.get("sample_id")}
                print(f"Loaded {len(processed_by_k[top_k])} processed samples for top_k={top_k} from {result_file}.")
        except FileNotFoundError:
            print(f"No existing results file for top_k={top_k}. Starting from scratch.")
    return results_by_k, processed_by_k


def summarize_retrieval_latency(sample_result):
    latencies = [qa.get("retrieval_latency_ms") for qa in sample_result.get("qa", [])]
    latencies = [lat for lat in latencies if isinstance(lat, (int, float))]
    if not latencies:
        return {"count": 0, "average_retrieval_latency_ms": 0.0}
    return {
        "count": len(latencies),
        "average_retrieval_latency_ms": sum(latencies) / len(latencies),
        "min_retrieval_latency_ms": min(latencies),
        "max_retrieval_latency_ms": max(latencies),
    }


def write_results_for_k(top_k, results, result_file):
    write_json_atomic(result_file, results)

    all_latencies = []
    for sample_result in results:
        for qa in sample_result.get("qa", []):
            latency = qa.get("retrieval_latency_ms")
            if isinstance(latency, (int, float)):
                all_latencies.append(latency)
    summary = {
        "top_k": top_k,
        "count": len(all_latencies),
        "average_retrieval_latency_ms": sum(all_latencies) / len(all_latencies) if all_latencies else 0.0,
        "min_retrieval_latency_ms": min(all_latencies) if all_latencies else 0.0,
        "max_retrieval_latency_ms": max(all_latencies) if all_latencies else 0.0,
        "result_file": result_file,
    }
    summary_path = os.path.join(os.path.dirname(result_file), "retrieval_latency_summary.json")
    write_json_atomic(summary_path, summary)


def run_retrieval(history, memo, bot_thinking, prompts, client, openai_modelid, encoding, retrieve_top_k=None):
    topics = []
    for k, v in memo.items():
        for vv in v:
            topics.append((k, vv["summary"], vv["dialogs"]))
    system_insturction = prompts["retrieval"]["system"]
    task_instruction = prompts["retrieval"]["instruction"]
    if retrieve_top_k is not None:
        task_instruction = task_instruction.replace(
            "Select one or more topics",
            f"Select at most {retrieve_top_k} topics",
        )
        task_instruction += f"\nReturn no more than {retrieve_top_k} option numbers."
    task_case = "```\nQuery Sentence:\n" + history["User Input"][6:] + "\nTopic Options:\n" + \
                "\n".join(["({}) {}".format(v_i + 1, v[0] + ". " + v[1]) for v_i, v in enumerate(topics)]) + "\n```"
    qs = q_pre + system_insturction.replace("OPTION", str(len(topics))) + task_case + task_instruction.replace("OPTION", str(len(topics))) + qa_link
    outputs = gen_model_output(qs, "retrieval_dialogsum", client, openai_modelid, encoding)
    if outputs is None:
        outputs = ""
    outputs = outputs.split("#")
    chosen_topics = []
    seen_indexes = set()
    for output in outputs:
        try:
            index_ = int(output.strip()) - 1
        except Exception:
            continue
        if index_ in seen_indexes:
            continue
        seen_indexes.add(index_)
        if 0 <= index_ < len(topics) and topics[index_][0] != "NOTO":
            chosen_topics.append(topics[index_])
        if retrieve_top_k is not None and len(chosen_topics) >= retrieve_top_k:
            break
    if len(chosen_topics) > 0:
        history["Related Topics"] = [ct[0] for ct in chosen_topics]
        history["Related Summaries"] = [ct[1] for ct in chosen_topics]
        history["Related Dialogs"] = [" ### ".join(ct[2]) for ct in chosen_topics]
    else:
        history["Related Topics"] = []
        history["Related Summaries"] = []
        history["Related Dialogs"] = []
    bot_thinking["retrieval"] = {"input": qs, "output": outputs, "top_k": retrieve_top_k}
    return history, bot_thinking


def answer_one_qa(qa_item, memo, prompts, client, openai_modelid, encoding, retrieve_top_k):
    question = qa_item["question"]
    history = {
        "Recent Dialogs": [],
        "Related Topics": [],
        "Related Summaries": [],
        "Related Dialogs": [],
        "User Input": "user: " + question,
    }
    bot_thinking = {"retrieval": "", "summarization": ""}
    retrieval_latency_ms = 0.0

    if len(memo.keys()) > 1:
        retrieval_start = time.perf_counter()
        history, bot_thinking = run_retrieval(
            history,
            memo,
            bot_thinking,
            prompts,
            client,
            openai_modelid,
            encoding,
            retrieve_top_k=retrieve_top_k,
        )
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        system_instruction = prompts["chatting"]["system"]
        task_instruction = prompts["chatting"]["instruction"]
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

    retrieved = history["Related Summaries"] + history["Related Dialogs"]

    category = qa_item.get("category", 1)
    return {
        "question": question,
        "answer": qa_item.get("answer"),
        "category": category,
        "response": answer,
        "retrieved": retrieved,
        "retrieved_count": len(history["Related Topics"]),
        "retrieval_top_k": retrieve_top_k,
        "retrieval_latency_ms": retrieval_latency_ms,
        "search_duration_ms": retrieval_latency_ms,
        "retrieval_trace": bot_thinking.get("retrieval", {}),
    }


_MEMOCHAT_WORKER_STATE = {}


def _init_memochat_worker(prompts, openai_modelid, base_url, api_key, qa_concurrency):
    global _MEMOCHAT_WORKER_STATE
    _MEMOCHAT_WORKER_STATE = {
        "prompts": prompts,
        "openai_modelid": openai_modelid,
        "client": OpenAI(api_key=api_key, base_url=base_url),
        "encoding": build_tokenizer(openai_modelid),
        "qa_concurrency": max(1, int(qa_concurrency or 1)),
    }


def _process_memochat_sample(task):
    d, pending_top_ks = task
    state = _MEMOCHAT_WORKER_STATE
    prompts = state["prompts"]
    openai_modelid = state["openai_modelid"]
    client = state["client"]
    encoding = state["encoding"]
    qa_concurrency = state["qa_concurrency"]
    sample_id = d["sample_id"]

    print("[{}] Starting index build.".format(sample_id), flush=True)
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

    conversation_data = d["conversation"]
    session_keys = natural_session_keys(conversation_data)

    for session_index, session_key in enumerate(session_keys, start=1):
        session_dialogs = conversation_data[session_key]
        session_time = conversation_data.get(session_key + "_date_time", "Unknown time")

        for dialog in session_dialogs:
            dialog_line = "{}: {} [Time: {}, ID: {}]".format(
                dialog["speaker"],
                format_turn_text(dialog),
                session_time,
                dialog.get("dia_id", ""),
            )
            history["Recent Dialogs"].append(dialog_line)

            if (
                len(" ### ".join(history["Recent Dialogs"]).split(" ")) > (MaxLen // 2)
                or len(history["Recent Dialogs"]) >= 10
            ):
                bot_thinking = {"retrieval": "", "summarization": ""}
                history, memo, bot_thinking = run_summary(
                    history,
                    memo,
                    bot_thinking,
                    prompts,
                    client,
                    openai_modelid,
                    encoding,
                )

        if session_index % 10 == 0:
            print(
                "[{}] Indexed {}/{} sessions; memo topics={}".format(
                    sample_id, session_index, len(session_keys), len(memo)
                ),
                flush=True,
            )

    if history["Recent Dialogs"]:
        bot_thinking = {"retrieval": "", "summarization": ""}
        history, memo, bot_thinking = run_summary(
            history,
            memo,
            bot_thinking,
            prompts,
            client,
            openai_modelid,
            encoding,
        )

    qa_items = d.get("qa", [])
    sample_results = {}
    for top_k in pending_top_ks:
        sample_result = {
            "sample_id": sample_id,
            "top_k": top_k,
            "qa": [],
        }

        if qa_concurrency == 1 or len(qa_items) <= 1:
            sample_result["qa"] = [
                answer_one_qa(
                    qa_item,
                    memo,
                    prompts,
                    client,
                    openai_modelid,
                    encoding,
                    top_k,
                )
                for qa_item in qa_items
            ]
        else:
            ordered_results = [None] * len(qa_items)
            with ThreadPoolExecutor(max_workers=qa_concurrency) as executor:
                future_to_index = {
                    executor.submit(
                        answer_one_qa,
                        qa_item,
                        memo,
                        prompts,
                        client,
                        openai_modelid,
                        encoding,
                        top_k,
                    ): index
                    for index, qa_item in enumerate(qa_items)
                }
                for future in as_completed(future_to_index):
                    ordered_results[future_to_index[future]] = future.result()
            sample_result["qa"] = ordered_results

        sample_result["retrieval_latency_summary"] = summarize_retrieval_latency(sample_result)
        sample_results[top_k] = sample_result

    print(
        "[{}] Finished; sessions={}, memo topics={}, qa={}".format(
            sample_id, len(session_keys), len(memo), len(qa_items)
        ),
        flush=True,
    )
    return {
        "sample_id": sample_id,
        "memory": {"sample_id": sample_id, "memo": memo},
        "results": sample_results,
    }


def _run_memochat_parallel(
    input_data,
    results_output_path,
    memory_output_path,
    prompt_path,
    openai_modelid,
    base_url,
    api_key,
    top_ks,
    qa_concurrency,
    sample_concurrency,
):
    result_files, memory_file = resolve_output_files(
        results_output_path, memory_output_path, top_ks
    )
    prompts = load_json_file(prompt_path)
    data = load_json_file(input_data)

    for result_file in result_files.values():
        ensure_parent_dir(result_file)
    ensure_parent_dir(memory_file)

    results_by_k, processed_by_k = load_existing_results(result_files)
    try:
        memory = load_json_file(memory_file)
        if not isinstance(memory, list):
            memory = []
    except FileNotFoundError:
        memory = []

    memory_by_id = {
        item["sample_id"]: item
        for item in memory
        if isinstance(item, dict) and item.get("sample_id")
    }
    tasks = []
    for d in data:
        sample_id = d["sample_id"]
        pending_top_ks = [
            top_k for top_k in top_ks if sample_id not in processed_by_k[top_k]
        ]
        if pending_top_ks:
            tasks.append((d, pending_top_ks))

    completed_before = len(data) - len(tasks)
    print("Starting sample-level multiprocessing.", flush=True)
    print("Sample concurrency: {}".format(sample_concurrency), flush=True)
    print(
        "Resume state: {}/{} samples complete; {} pending.".format(
            completed_before, len(data), len(tasks)
        ),
        flush=True,
    )
    for top_k, result_file in result_files.items():
        print("top_k={} result file: {}".format(top_k, result_file), flush=True)
    print("Memory file: {}".format(memory_file), flush=True)

    if not tasks:
        return results_by_k, [
            memory_by_id[d["sample_id"]]
            for d in data
            if d["sample_id"] in memory_by_id
        ]

    failures = []
    finished_now = 0
    with ProcessPoolExecutor(
        max_workers=sample_concurrency,
        initializer=_init_memochat_worker,
        initargs=(prompts, openai_modelid, base_url, api_key, qa_concurrency),
    ) as executor:
        future_to_sample_id = {
            executor.submit(_process_memochat_sample, task): task[0]["sample_id"]
            for task in tasks
        }
        for future in as_completed(future_to_sample_id):
            sample_id = future_to_sample_id[future]
            try:
                output = future.result()
            except Exception as exc:
                failures.append((sample_id, repr(exc)))
                print("[{}] FAILED: {!r}".format(sample_id, exc), flush=True)
                continue

            memory_by_id[sample_id] = output["memory"]
            ordered_memory = [
                memory_by_id[d["sample_id"]]
                for d in data
                if d["sample_id"] in memory_by_id
            ]
            write_json_atomic(memory_file, ordered_memory)

            for top_k, sample_result in output["results"].items():
                results_by_k[top_k].append(sample_result)
                processed_by_k[top_k].add(sample_id)
                write_results_for_k(top_k, results_by_k[top_k], result_files[top_k])

            finished_now += 1
            print(
                "Parallel progress: {}/{} complete; {}/{} finished this run.".format(
                    completed_before + finished_now,
                    len(data),
                    finished_now,
                    len(tasks),
                ),
                flush=True,
            )

    if failures:
        failure_text = ", ".join(
            "{}: {}".format(sample_id, error) for sample_id, error in failures
        )
        raise RuntimeError(
            "{} samples failed during multiprocessing: {}".format(
                len(failures), failure_text
            )
        )

    final_memory = [
        memory_by_id[d["sample_id"]]
        for d in data
        if d["sample_id"] in memory_by_id
    ]
    print("Processing finished.", flush=True)
    for top_k in top_ks:
        print(
            "top_k={}: total samples saved: {}".format(
                top_k, len(results_by_k[top_k])
            ),
            flush=True,
        )
    return results_by_k, final_memory


def run_memochat(input_data, results_output_path, memory_output_path, prompt_path,
                 openai_modelid=DEFAULT_MODEL_ID, base_url=DEFAULT_BASE_URL, api_key=DEFAULT_API_KEY,
                 token_file=DEFAULT_TOKEN_FILE, retrieve_top_ks=None, qa_concurrency=1,
                 sample_concurrency=1):
    """
    Run the MemoChat conversational memory pipeline.

    The memory index is built once per sample. The QA stage can then be replayed
    with multiple retrieval top-k values while reusing the same memo.
    """
    print("Starting MemoChat...")

    openai_modelid = openai_modelid or DEFAULT_MODEL_ID
    base_url = base_url or DEFAULT_BASE_URL
    api_key = api_key or DEFAULT_API_KEY
    token_file = token_file or os.path.join(
        os.path.dirname(os.path.abspath(results_output_path)),
        "token_tracker.json",
    )
    top_ks = parse_top_ks(retrieve_top_ks)
    qa_concurrency = max(1, int(qa_concurrency or 1))
    sample_concurrency = max(1, int(sample_concurrency or 1))

    if sample_concurrency > 1:
        return _run_memochat_parallel(
            input_data=input_data,
            results_output_path=results_output_path,
            memory_output_path=memory_output_path,
            prompt_path=prompt_path,
            openai_modelid=openai_modelid,
            base_url=base_url,
            api_key=api_key,
            top_ks=top_ks,
            qa_concurrency=qa_concurrency,
            sample_concurrency=sample_concurrency,
        )

    result_files, memory_file = resolve_output_files(results_output_path, memory_output_path, top_ks)

    print(f"Model: {openai_modelid}")
    print(f"Base URL: {base_url}")
    print(f"Top-K values: {top_ks}")
    print(f"QA concurrency: {qa_concurrency}")
    for top_k, result_file in result_files.items():
        print(f"top_k={top_k} result file: {result_file}")
    print(f"Memory file: {memory_file}")

    encoding = build_tokenizer(openai_modelid)
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompts = load_json_file(prompt_path)
    data = load_json_file(input_data)

    for result_file in result_files.values():
        ensure_parent_dir(result_file)
    ensure_parent_dir(memory_file)

    results_by_k, processed_by_k = load_existing_results(result_files)
    memory = []
    try:
        with open(memory_file, "r", encoding="utf-8") as f:
            memory = json.load(f)
    except FileNotFoundError:
        memory = []

    for d in data:
        sample_id = d["sample_id"]
        if all(sample_id in processed_by_k[top_k] for top_k in top_ks):
            print(f"Skipping fully processed sample: {sample_id}")
            continue

        print("=" * 20 + "start of conversation {}".format(sample_id) + "=" * 20)
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

        conversation_data = d["conversation"]
        session_keys = natural_session_keys(conversation_data)

        for session_key in session_keys:
            print("Processing {}...".format(session_key))
            session_dialogs = conversation_data[session_key]
            session_time_key = session_key + "_date_time"
            session_time = conversation_data.get(session_time_key, "Unknown time")
            print("Session time: {}".format(session_time))

            for dial_id, dialog in enumerate(session_dialogs):
                speaker = dialog["speaker"]
                text = format_turn_text(dialog)
                dia_id = dialog.get("dia_id", "")
                dialog_line = "{}: {} [Time: {}, ID: {}]".format(speaker, text, session_time, dia_id)
                history["Recent Dialogs"].append(dialog_line)

                if len(" ### ".join(history["Recent Dialogs"]).split(" ")) > (MaxLen // 2) or len(history["Recent Dialogs"]) >= 10:
                    print("Summarization threshold reached. Starting summarization...")
                    bot_thinking = {"retrieval": "", "summarization": ""}
                    history, memo, bot_thinking = run_summary(history, memo, bot_thinking, prompts, client, openai_modelid, encoding)
                    print("Summarization finished. Current memo topics: {}".format(list(memo.keys())))

        if len(history["Recent Dialogs"]) > 0:
            print("Conversation finished. Running final summarization...")
            bot_thinking = {"retrieval": "", "summarization": ""}
            history, memo, bot_thinking = run_summary(history, memo, bot_thinking, prompts, client, openai_modelid, encoding)
            print("Final summarization finished.")

        memory.append({"sample_id": sample_id, "memo": memo})
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)

        qa_items = d.get("qa", [])
        for top_k in top_ks:
            if sample_id in processed_by_k[top_k]:
                print(f"Skipping processed sample {sample_id} for top_k={top_k}")
                continue

            print(f"Starting QA answering for top_k={top_k}...")
            sample_result = {
                "sample_id": sample_id,
                "top_k": top_k,
                "qa": []
            }

            if qa_concurrency == 1:
                for qa_item in qa_items:
                    print("Question: {}".format(qa_item["question"]))
                    qa_result = answer_one_qa(qa_item, memo, prompts, client, openai_modelid, encoding, top_k)
                    sample_result["qa"].append(qa_result)
                    print("Answer: {}".format(qa_result["response"]))
                    print("Retrieval latency ms: {:.2f}".format(qa_result["retrieval_latency_ms"]))
                    print("-" * 50)
            else:
                ordered_results = [None] * len(qa_items)
                with ThreadPoolExecutor(max_workers=qa_concurrency) as executor:
                    future_to_index = {
                        executor.submit(answer_one_qa, qa_item, memo, prompts, client, openai_modelid, encoding, top_k): idx
                        for idx, qa_item in enumerate(qa_items)
                    }
                    for future in as_completed(future_to_index):
                        idx = future_to_index[future]
                        ordered_results[idx] = future.result()
                        print(
                            "QA {}/{} for top_k={} finished, retrieval latency ms: {:.2f}".format(
                                idx + 1,
                                len(qa_items),
                                top_k,
                                ordered_results[idx]["retrieval_latency_ms"],
                            )
                        )
                sample_result["qa"] = ordered_results

            sample_result["retrieval_latency_summary"] = summarize_retrieval_latency(sample_result)
            results_by_k[top_k].append(sample_result)
            processed_by_k[top_k].add(sample_id)
            write_results_for_k(top_k, results_by_k[top_k], result_files[top_k])
            print(f"Saved top_k={top_k} results to disk.")

        print("=" * 20 + "end of conversation {}".format(sample_id) + "=" * 20)
        print("Final memo topics: {}".format(list(memo.keys())))
        print("QA count: {}".format(len(qa_items)))
        print("\n")

    print("Processing finished.")
    for top_k in top_ks:
        print(f"top_k={top_k}: total samples saved: {len(results_by_k[top_k])}")
    return results_by_k, memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MemoChat on a benchmark dataset.")
    parser.add_argument("--input-data", required=True)
    parser.add_argument("--results-output-path", required=True)
    parser.add_argument("--memory-output-path")
    parser.add_argument(
        "--prompt-path",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_loco.json"),
    )
    parser.add_argument("--retrieve-top-ks", default="10")
    parser.add_argument("--token-file")
    args = parser.parse_args()
    run_memochat(
        input_data=args.input_data,
        results_output_path=args.results_output_path,
        memory_output_path=args.memory_output_path,
        prompt_path=args.prompt_path,
        token_file=args.token_file,
        retrieve_top_ks=args.retrieve_top_ks,
    )
