import argparse
import asyncio
import json

from time import time

import pandas as pd

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

from .configuration import ensure_parent_dir
from .prompts import ANSWER_PROMPT_MEMOS


async def locomo_response(llm_client, model_name: str, context: str, question: str) -> str:
    prompt = ANSWER_PROMPT_MEMOS.format(
        context=context,
        question=question,
    )
    response = await llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
        ],
        temperature=0,
    )
    result = response.choices[0].message.content or ""
    return result


async def process_qa(qa, search_result, oai_client, model_name):
    start = time()
    query = qa.get("question")
    gold_answer = qa.get("answer")
    qa_category = qa.get("category")

    answer = await locomo_response(oai_client, model_name, search_result.get("context"), query)
    response_duration_ms = (time() - start) * 1000

    print(f"Processed question: {query}")
    print(f"Answer: {answer}")
    return {
        "question": query,
        "answer": answer,
        "category": qa_category,
        "golden_answer": gold_answer,
        "search_context": search_result.get("context", ""),
        "response_duration_ms": response_duration_ms,
        "search_duration_ms": search_result.get("duration_ms", 0),
    }


def save_formatted_results(all_responses, locomo_df, runtime_config):
    formatted_results = []

    num_users = locomo_df.shape[0]
    for group_idx in range(num_users):
        sample_data = locomo_df.iloc[group_idx]
        sample_id = sample_data.get("sample_id", f"sample_{group_idx}")

        group_id = f"locomo_exp_user_{group_idx}"
        group_responses = all_responses.get(group_id, [])

        formatted_qa = []
        for response_data in group_responses:
            qa_item = {
                "question": response_data.get("question", ""),
                "answer": response_data.get("golden_answer", ""),
                "category": response_data.get("category", 0),
                "response": response_data.get("answer", ""),
                "retrieved": [response_data.get("search_context", "")],
            }
            formatted_qa.append(qa_item)

        sample_result = {
            "sample_id": sample_id,
            "qa": formatted_qa,
        }
        formatted_results.append(sample_result)

    result_path = runtime_config["formatted_results_path"]
    ensure_parent_dir(result_path)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(formatted_results, f, indent=4)
        print(f"Saved formatted results to {result_path}")


async def response(runtime_config):
    search_path = runtime_config["search_results_path"]
    response_path = runtime_config["response_results_path"]

    load_dotenv()
    oai_client = AsyncOpenAI(
        api_key=runtime_config["response_api_key"],
        base_url=runtime_config["response_base_url"],
    )

    locomo_df = pd.read_json(runtime_config["dataset_path"])
    with open(search_path, "r", encoding="utf-8") as file:
        locomo_search_results = json.load(file)

    num_users = locomo_df.shape[0]

    all_responses = {}
    for group_idx in range(num_users):
        qa_set = locomo_df["qa"].iloc[group_idx]
        qa_set_filtered = [qa for qa in qa_set if qa.get("category") != 5]

        group_id = f"locomo_exp_user_{group_idx}"
        search_results = locomo_search_results.get(group_id)

        matched_pairs = []
        for qa in qa_set_filtered:
            question = qa.get("question")
            matching_result = next(
                (result for result in search_results if result.get("query") == question), None
            )
            if matching_result:
                matched_pairs.append((qa, matching_result))
            else:
                print(f"Warning: No matching search result found for question: {question}")

        tasks = [
            process_qa(qa, search_result, oai_client, runtime_config["response_model"])
            for qa, search_result in tqdm(
                matched_pairs,
                desc=f"Processing {group_id}",
                total=len(matched_pairs),
            )
        ]

        responses = await asyncio.gather(*tasks)
        all_responses[group_id] = responses

    print(all_responses)

    ensure_parent_dir(response_path)
    with open(response_path, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, indent=2)
        print("Save response results")

    save_formatted_results(all_responses, locomo_df, runtime_config)


if __name__ == "__main__":
    from .configuration import build_runtime_config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version identifier for loading results (e.g., 1010)",
    )
    args = parser.parse_args()

    runtime_config = build_runtime_config({"version": args.version})
    asyncio.run(response(runtime_config))
