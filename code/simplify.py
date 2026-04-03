import argparse
import asyncio
import json
import os

import aiofiles
from openai import AsyncOpenAI


DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL_NAME = "Qwen2.5-7B-Instruct"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 150
DEFAULT_INPUT_FILENAME = "result.json"
DEFAULT_OUTPUT_FILENAME = "result_simplified.json"

DATASET_CONFIGS = {
    "loco": {
        "display_name": "LOCOMO",
        "result_root": "Result/LOCOMO",
        "method_choices": [
            "amem",
            "memochat",
            "memoryos",
            "memos",
            "memtree",
            "zep",
            "mem0",
            "mem0g",
            "memorybank",
            "memgpt",
            "sota",
        ],
    },
    "lme": {
        "display_name": "LONGMEMEVAL",
        "result_root": "Result/LONGMEMEVAL",
        "method_choices": [
            "amem",
            "memochat",
            "memoryos",
            "memos",
            "memtree",
            "zep",
            "mem0",
            "mem0g",
            "memorybank",
            "sota",
            "memgpt",
        ],
    },
}

PROMPT = """Your task is to act as an answer simplifier. I will give you a question and a full-sentence answer. You must reduce the answer to its most critical component.
Follow these rules:
1. **Extract the Core Information:** Identify the primary piece of information that directly answers the question.
2. **Remove Extraneous Phrases:** Eliminate phrases like "Based on the information provided...", "The answer is...", and "As per the document...".
3. **Omit Explanations:** Do not include justifications, reasoning, or additional context from the original answer.
4. **Be Concise:** The output should be the shortest possible string that still accurately answers the question.

Example:
Question: "What degree did I graduate with?"
Original Answer: "Based on the information provided, you graduated with a degree in Business Administration."
Simplified Answer: "Business Administration"

Here is the question and answer to simplify:
Question: {question}
Original Answer: {answer}
Simplified Answer:
"""


def build_client(api_key, base_url):
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def build_default_output_path(input_file):
    root, ext = os.path.splitext(input_file)
    if ext:
        return f"{root}_simplified{ext}"
    return f"{input_file}_simplified.json"


def build_results_dir(current_dir, dataset, method, version):
    results_dir = os.path.join(current_dir, DATASET_CONFIGS[dataset]["result_root"], method)
    if version:
        results_dir = os.path.join(results_dir, version)
    return results_dir


def resolve_file_paths(input_file, output_file, dataset, method, version):
    if input_file:
        return input_file, output_file or build_default_output_path(input_file)

    if not dataset or not method:
        raise ValueError(
            "Either provide --input_file, or provide both --dataset and --method."
        )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = build_results_dir(current_dir, dataset, method, version)
    resolved_input = os.path.join(results_dir, DEFAULT_INPUT_FILENAME)
    resolved_output = output_file or os.path.join(results_dir, DEFAULT_OUTPUT_FILENAME)
    return resolved_input, resolved_output


async def simplify_response(
    question,
    response,
    client,
    model_name,
    temperature,
    max_tokens,
):
    if not response or response.strip() == "":
        return response

    try:
        prompt = PROMPT.format(question=question, answer=response)
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        simplified = completion.choices[0].message.content.strip()
        return simplified
    except Exception as e:
        print(f"Error simplifying response for question '{question}': {e}")
        return response


async def process_qa_batch(
    qas,
    sample_id,
    client,
    model_name,
    temperature,
    max_tokens,
):
    tasks = []
    qa_indices = []

    for i, qa in enumerate(qas):
        if "response" in qa and qa["response"]:
            task = simplify_response(
                qa["question"],
                qa["response"],
                client,
                model_name,
                temperature,
                max_tokens,
            )
            tasks.append(task)
            qa_indices.append(i)

    if tasks:
        simplified_results = await asyncio.gather(*tasks)
        for idx, simplified in zip(qa_indices, simplified_results):
            simplified = simplified.strip("\"")
            qas[idx]["response"] = simplified
            print(f"Sample {sample_id} - Question: {qas[idx]['question'][:50]}... -> Simplified")

    return qas


async def process_json_file(
    input_file,
    output_file,
    client,
    model_name,
    temperature,
    max_tokens,
):
    print(f"Starting file processing: {input_file}")

    async with aiofiles.open(input_file, "r", encoding="utf-8") as f:
        content = await f.read()
        data = json.loads(content)

    print(f"Loaded {len(data)} samples")

    processed_data = []
    for i, sample in enumerate(data):
        sample_id = sample.get("sample_id", f"sample_{i}")
        print(f"Processing sample {i + 1}/{len(data)}: {sample_id}")

        if "qa" in sample:
            processed_qa = await process_qa_batch(
                sample["qa"],
                sample_id,
                client,
                model_name,
                temperature,
                max_tokens,
            )
            sample["qa"] = processed_qa

        processed_data.append(sample)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    async with aiofiles.open(output_file, "w", encoding="utf-8") as f:
        await f.write(json.dumps(processed_data, ensure_ascii=False, indent=2))

    print(f"Processing finished. Results saved to: {output_file}")


async def main():
    parser = argparse.ArgumentParser(description="Simplify QA responses in a result JSON file.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Dataset profile used to resolve the default result path.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method name under the result directory when using dataset-based path resolution.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Optional version subdirectory under the method result directory.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to the source result JSON file. If omitted, it is derived from dataset/method/version.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to the simplified result JSON file. Defaults to '<input>_simplified.json'.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key for the OpenAI-compatible endpoint. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help="Base URL for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name used for response simplification.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for the simplification model.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens returned by the simplification model.",
    )
    args = parser.parse_args()

    if args.method and not args.dataset:
        parser.error("--dataset is required when --method is provided.")

    if args.dataset and args.method:
        valid_methods = DATASET_CONFIGS[args.dataset]["method_choices"]
        if args.method not in valid_methods:
            parser.error(
                f"Invalid --method '{args.method}' for dataset '{args.dataset}'. "
                f"Valid choices: {valid_methods}"
            )

    try:
        input_file, output_file = resolve_file_paths(
            args.input_file,
            args.output_file,
            args.dataset,
            args.method,
            args.version,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Using input file: {input_file}")
    print(f"Using output file: {output_file}")
    client = build_client(args.api_key, args.base_url)

    try:
        await process_json_file(
            input_file,
            output_file,
            client,
            args.model,
            args.temperature,
            args.max_tokens,
        )
        print("All processing completed.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
