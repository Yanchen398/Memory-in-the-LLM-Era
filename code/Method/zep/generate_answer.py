import asyncio
import json
import os
from typing import Any, Dict

from openai import AsyncOpenAI


DEFAULT_ANSWER_MODEL = "/home/docker/LLaMA-Factory/output/qwen2_5_lora_sft"
DEFAULT_ANSWER_API_KEY = "empty"
DEFAULT_ANSWER_BASE_URL = "http://localhost:8000/v1"


def ensure_parent_dir(path):
    if path:
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)


def build_answer_context(retrieved: Any) -> str:
    sections = []
    for section_name in ("Edges", "Nodes", "Episodes", "Communities"):
        values = retrieved.get(section_name, []) if isinstance(retrieved, dict) else []
        if values:
            sections.append(f"[{section_name}]\n" + "\n".join(values))
    return "\n\n".join(sections)


async def generate_answer(
    question: str,
    retrieved: Any,
    model: str = DEFAULT_ANSWER_MODEL,
    api_key: str = DEFAULT_ANSWER_API_KEY,
    base_url: str = DEFAULT_ANSWER_BASE_URL,
) -> str:
    try:
        context = build_answer_context(retrieved)
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful expert assistant answering questions from retrieved conversation context.",
                },
                {
                    "role": "user",
                    "content": (
                        "# CONTEXT\n"
                        f"{context}\n\n"
                        "# QUESTION\n"
                        f"{question}\n\n"
                        "Answer briefly and directly based only on the context."
                    ),
                },
            ],
            max_tokens=200,
            temperature=0,
        )
        return response.choices[0].message.content.strip() if response.choices[0].message.content else "No response generated"
    except Exception as e:
        print(f"Error generating answer for question '{question}': {e}")
        return "Error: Unable to generate answer"


async def process_qa_item(
    qa_item: Dict[str, Any],
    model: str = DEFAULT_ANSWER_MODEL,
    api_key: str = DEFAULT_ANSWER_API_KEY,
    base_url: str = DEFAULT_ANSWER_BASE_URL,
) -> Dict[str, Any]:
    question = qa_item.get("question", "")
    retrieved = qa_item.get("retrieved", "")

    if question and retrieved:
        qa_item["response"] = await generate_answer(
            question,
            retrieved,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        print(f"Processed question: {question[:50]}...")
    else:
        qa_item["response"] = "No question or retrieved memories available"
        print("Skipped item due to missing question or memories")

    return qa_item


async def process_sample(
    sample: Dict[str, Any],
    model: str = DEFAULT_ANSWER_MODEL,
    api_key: str = DEFAULT_ANSWER_API_KEY,
    base_url: str = DEFAULT_ANSWER_BASE_URL,
) -> Dict[str, Any]:
    sample_id = sample.get("sample_id", "unknown")
    qa_list = sample.get("qa", [])

    print(f"Processing sample: {sample_id} with {len(qa_list)} Q&A items")
    tasks = [
        process_qa_item(qa_item, model=model, api_key=api_key, base_url=base_url)
        for qa_item in qa_list
    ]
    updated_qa_list = await asyncio.gather(*tasks, return_exceptions=True)

    final_qa_list = []
    for i, result in enumerate(updated_qa_list):
        if isinstance(result, Exception):
            print(f"Error processing Q&A item {i + 1}: {result}")
            qa_item = qa_list[i].copy()
            qa_item["response"] = f"Error: {result}"
            final_qa_list.append(qa_item)
        else:
            final_qa_list.append(result)
            print(f"  Completed Q&A {i + 1}/{len(qa_list)}")

    sample["qa"] = final_qa_list
    return sample


async def main():
    input_file = "/home/docker/IndepthMem/Result/LOCOMO/zep/result_raw.json"
    output_file = input_file.replace("result_raw.json", "result.json")

    try:
        print(f"Loading data from {input_file}...")
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Loaded {len(data)} samples")
        updated_data = []
        for i, sample in enumerate(data):
            try:
                print(f"\nProcessing sample {i + 1}/{len(data)}")
                updated_sample = await process_sample(sample)
                updated_data.append(updated_sample)
            except Exception as e:
                print(f"Error processing sample {i + 1}: {e}")
                updated_data.append(sample)

        ensure_parent_dir(output_file)
        print(f"\nSaving results to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(updated_data, f, indent=2, ensure_ascii=False)

        print(f"Successfully processed all samples and saved to {output_file}")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {input_file}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
