import argparse
import json
import os

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

import pandas as pd

from dotenv import load_dotenv
from tqdm import tqdm

from memos.configs.mem_os import MOSConfig
from memos.mem_os.main import MOS

from .configuration import (
    build_mos_config,
    ensure_dir,
    ensure_parent_dir,
    get_storage_path,
    get_tmp_search_results_path,
)
from .utils import filter_memory_data


def get_client(user_id: str, runtime_config, top_k: int = 20):
    mos_config_data = build_mos_config(runtime_config, top_k=top_k)
    mos_config = MOSConfig(**mos_config_data)
    mos = MOS(mos_config)
    mos.create_user(user_id=user_id)

    storage_path = get_storage_path(runtime_config, user_id)
    mos.register_mem_cube(
        mem_cube_name_or_path=storage_path,
        mem_cube_id=user_id,
        user_id=user_id,
    )

    return mos


TEMPLATE_MEMOS = """Memories for user {speaker_1}:

    {speaker_1_memories}

    Memories for user {speaker_2}:

    {speaker_2_memories}
"""


def memos_search(client, query, conv_id, speaker_a, speaker_b, reversed_client):
    start = time()
    search_a_results = client.search(
        query=query,
        user_id=conv_id + "_speaker_a",
    )
    filtered_search_a_results = filter_memory_data(search_a_results)["text_mem"][0]["memories"]
    speaker_a_context = ""
    for item in filtered_search_a_results:
        speaker_a_context += f"{item['memory']}\n"

    search_b_results = reversed_client.search(
        query=query,
        user_id=conv_id + "_speaker_b",
    )
    filtered_search_b_results = filter_memory_data(search_b_results)["text_mem"][0]["memories"]
    speaker_b_context = ""
    for item in filtered_search_b_results:
        speaker_b_context += f"{item['memory']}\n"

    context = TEMPLATE_MEMOS.format(
        speaker_1=speaker_a,
        speaker_1_memories=speaker_a_context,
        speaker_2=speaker_b,
        speaker_2_memories=speaker_b_context,
    )

    print(query, context)
    duration_ms = (time() - start) * 1000
    return context, duration_ms


def search_query(client, query, metadata, reversed_client=None):
    conv_id = metadata.get("conv_id")
    speaker_a = metadata.get("speaker_a")
    speaker_b = metadata.get("speaker_b")

    context, duration_ms = memos_search(
        client, query, conv_id, speaker_a, speaker_b, reversed_client
    )
    return context, duration_ms


def load_existing_results(runtime_config, group_idx):
    result_path = get_tmp_search_results_path(runtime_config, group_idx)
    if os.path.exists(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                return json.load(f), True
        except Exception as e:
            print(f"Error loading existing results for group {group_idx}: {e}")
    return {}, False


def process_user(group_idx, locomo_df, runtime_config, top_k=20, num_workers=1):
    search_results = defaultdict(list)
    qa_set = locomo_df["qa"].iloc[group_idx]
    conversation = locomo_df["conversation"].iloc[group_idx]
    speaker_a = conversation.get("speaker_a")
    speaker_b = conversation.get("speaker_b")
    speaker_a_user_id = f"{speaker_a}_{group_idx}"
    speaker_b_user_id = f"{speaker_b}_{group_idx}"
    conv_id = f"locomo_exp_user_{group_idx}"

    existing_results, loaded = load_existing_results(runtime_config, group_idx)
    if loaded:
        print(f"Loaded existing results for group {group_idx}")
        return existing_results

    metadata = {
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "speaker_a_user_id": speaker_a_user_id,
        "speaker_b_user_id": speaker_b_user_id,
        "conv_idx": group_idx,
        "conv_id": conv_id,
    }

    speaker_a_user_id = conv_id + "_speaker_a"
    speaker_b_user_id = conv_id + "_speaker_b"
    client = get_client(speaker_a_user_id, runtime_config, top_k=top_k)
    reversed_client = get_client(speaker_b_user_id, runtime_config, top_k=top_k)

    def process_qa(qa):
        query = qa.get("question")
        if qa.get("category") == 5:
            return None
        context, duration_ms = search_query(
            client, query, metadata, reversed_client=reversed_client
        )

        if not context:
            print(f"No context found for query: {query}")
            context = ""
        return {"query": query, "context": context, "duration_ms": duration_ms}

    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for qa in qa_set:
            futures.append(executor.submit(process_qa, qa))

        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"Processing user {group_idx}"
        ):
            result = future.result()
            if result:
                context_preview = (
                    result["context"][:20] + "..." if result["context"] else "No context"
                )
                print(
                    {
                        "query": result["query"],
                        "context": context_preview,
                        "duration_ms": result["duration_ms"],
                    }
                )
                search_results[conv_id].append(result)

    ensure_dir(runtime_config["tmp_dir"])
    tmp_result_path = get_tmp_search_results_path(runtime_config, group_idx)
    with open(tmp_result_path, "w", encoding="utf-8") as f:
        json.dump(dict(search_results), f, indent=2)
        print(f"Save search results {group_idx}")

    return search_results


def search(runtime_config):
    load_dotenv()
    locomo_df = pd.read_json(runtime_config["dataset_path"])

    num_users = locomo_df.shape[0]
    ensure_dir(runtime_config["result_dir"])
    all_search_results = defaultdict(list)

    for idx in range(num_users):
        try:
            print(f"Processing user {idx}...")
            user_results = process_user(
                idx,
                locomo_df,
                runtime_config,
                runtime_config["top_k"],
                runtime_config["num_workers"],
            )
            for conv_id, results in user_results.items():
                all_search_results[conv_id].extend(results)
        except Exception as e:
            print(f"User {idx} generated an exception: {e}")

    ensure_parent_dir(runtime_config["search_results_path"])
    with open(runtime_config["search_results_path"], "w", encoding="utf-8") as f:
        json.dump(dict(all_search_results), f, indent=2)
        print("Save all search results")


if __name__ == "__main__":
    from .configuration import build_runtime_config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version identifier for saving results (e.g., 1010)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers to process users"
    )
    parser.add_argument(
        "--top_k", type=int, default=20, help="Number of results to retrieve in search queries"
    )
    args = parser.parse_args()

    runtime_config = build_runtime_config(
        {
            "version": args.version,
            "num_workers": args.workers,
            "top_k": args.top_k,
        }
    )
    search(runtime_config)
