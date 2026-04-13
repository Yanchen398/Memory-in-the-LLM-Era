import argparse
import atexit
import copy
import glob
import json
import os
import re
from typing import Optional

try:
    from .load_dataset import load_locomo_dataset
    from .memory_layer import AgenticMemorySystem, LLMController
except ImportError:
    from load_dataset import load_locomo_dataset
    from memory_layer import AgenticMemorySystem, LLMController

from Method.memoryos.token_tracker import TokenTracker


DEFAULT_LLM_MODEL = "/home/docker/LLaMA-Factory/output/qwen2_5_lora_sft"
DEFAULT_LLM_API_KEY = "empty"
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def ensure_parent_dir(path: Optional[str]) -> None:
    if not path:
        return
    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def load_json_file(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def write_json_file(path: str, data) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return sanitized.strip("_") or "sample"


def create_stage_node(name: str):
    return {
        "name": name,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "start_time": None,
        "end_time": None,
        "duration_seconds": 0.0,
        "sub_stages": {},
    }


def aggregate_token_tree(node):
    aggregated = copy.deepcopy(node)
    total_prompt = aggregated.get("prompt_tokens", 0)
    total_completion = aggregated.get("completion_tokens", 0)
    total_tokens = aggregated.get("total_tokens", 0)

    sub_stages = aggregated.get("sub_stages", {})
    for name, sub_node in list(sub_stages.items()):
        aggregated_sub_node = aggregate_token_tree(sub_node)
        sub_stages[name] = aggregated_sub_node
        total_prompt += aggregated_sub_node.get("prompt_tokens", 0)
        total_completion += aggregated_sub_node.get("completion_tokens", 0)
        total_tokens += aggregated_sub_node.get("total_tokens", 0)

    aggregated["prompt_tokens"] = total_prompt
    aggregated["completion_tokens"] = total_completion
    aggregated["total_tokens"] = total_tokens
    return aggregated


def write_sample_token_tree(tracker: TokenTracker, sample_stage_name: str, sample_token_path: str) -> None:
    sample_node = tracker.root.get("sub_stages", {}).get(sample_stage_name)
    if sample_node is None:
        return
    write_json_file(sample_token_path, aggregate_token_tree(sample_node))


def rebuild_aggregate_token_file(output_dir: str, token_file: str) -> None:
    root = create_stage_node("root")
    pattern = os.path.join(output_dir, "sample_*", "token_stat.json")
    for sample_token_path in sorted(glob.glob(pattern)):
        sample_data = load_json_file(sample_token_path, None)
        if not sample_data:
            continue
        stage_name = sample_data.get("name") or os.path.basename(os.path.dirname(sample_token_path))
        root["sub_stages"][stage_name] = sample_data
        root["prompt_tokens"] += sample_data.get("prompt_tokens", 0)
        root["completion_tokens"] += sample_data.get("completion_tokens", 0)
        root["total_tokens"] += sample_data.get("total_tokens", 0)
    write_json_file(token_file, root)


class SimpleMemAgent:
    def __init__(
        self,
        model,
        backend,
        retrieve_k,
        api_key=DEFAULT_LLM_API_KEY,
        base_url=DEFAULT_LLM_BASE_URL,
        embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
    ):
        self.memory_system = AgenticMemorySystem(
            model_name=embedding_model_name,
            llm_backend=backend,
            llm_model=model,
            api_key=api_key,
            base_url=base_url,
        )
        self.retriever_llm = LLMController(
            backend=backend,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        self.retrieve_k = retrieve_k

    def add_memory(self, content, time=None):
        try:
            self.memory_system.add_note(content, time=time)
        except Exception as e:
            print(f"Warning: failed to add memory entry: {e}")
            return None

    def retrieve_memory(self, content, k=10):
        return self.memory_system.find_related_memories_raw(content, k=k)

    def generate_query_llm(self, question):
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

Question: {question}

Format your response as a JSON object with a "keywords" field containing the selected text.

Example response format:
{{"keywords": "keyword1, keyword2, keyword3"}}"""

        response = self.retriever_llm.llm.get_completion(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "keywords": {"type": "string"},
                        },
                        "required": ["keywords"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        )
        try:
            response = json.loads(response)["keywords"]
        except Exception:
            response = str(response).strip()
        return response

    def answer_question(self, question, category=None, answer="", return_retrieved=False):
        del category, answer

        keywords = self.generate_query_llm(question)
        raw_context = self.retrieve_memory(keywords, k=self.retrieve_k)

        if isinstance(raw_context, list):
            retrieved_memories = [str(memory) for memory in raw_context[: self.retrieve_k]]
            context = "\n".join(retrieved_memories)
        else:
            memories = str(raw_context).split("\n")
            retrieved_memories = memories[: self.retrieve_k]
            context = "\n".join(memories[: self.retrieve_k])

        user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.
Question: {question}
Short answer:
"""

        response = self.memory_system.llm_controller.llm.get_completion(
            user_prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                        },
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            temperature=0.7,
        )

        prediction = ""
        if response and str(response).strip():
            try:
                prediction = json.loads(response)["answer"]
            except Exception as e:
                print(f"Warning: failed to parse AMEM answer JSON: {e}")
                print(f"Raw response: {response}")

        if return_retrieved:
            return prediction, retrieved_memories
        return prediction


def simple_qa_session(
    dataset_path: str,
    model: str = DEFAULT_LLM_MODEL,
    modelname: Optional[str] = None,
    output_path: Optional[str] = None,
    ratio: float = 1.0,
    backend: str = "openai",
    retrieve_k: int = 10,
    use_cache: bool = False,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    api_key: str = DEFAULT_LLM_API_KEY,
    base_url: str = DEFAULT_LLM_BASE_URL,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    token_file: Optional[str] = None,
):
    del modelname, use_cache

    if output_path:
        output_path = os.path.abspath(output_path)
        output_dir = os.path.dirname(output_path)
    else:
        output_dir = os.path.join(os.path.dirname(__file__), "result")
        output_path = os.path.join(output_dir, "result.json")

    ensure_dir(output_dir)

    if not token_file:
        token_file = os.path.join(output_dir, "token_tracker.json")
    token_file = os.path.abspath(token_file)

    print(f"Loading dataset from {dataset_path}")
    samples = load_locomo_dataset(dataset_path)

    if end_idx is None:
        end_idx = len(samples)

    samples = samples[start_idx:end_idx]
    print(f"Processing samples from {start_idx} to {end_idx - 1}, total {len(samples)}")

    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        print(f"Using {num_samples} samples ({ratio * 100:.1f}% of dataset)")

    results = load_json_file(output_path, [])
    processed_sample_ids = {
        str(item.get("sample_id"))
        for item in results
        if isinstance(item, dict) and item.get("sample_id") is not None
    }

    runtime_token_file = os.path.join(output_dir, ".amem_token_tracker_runtime.json")
    tracker = TokenTracker(output_file=runtime_token_file)
    try:
        atexit.unregister(tracker.save_to_json)
    except Exception:
        pass
    tracker.patch_llm_api()

    total_questions = 0

    for sample_idx, sample in enumerate(samples):
        sample_id = str(sample.sample_id)
        if sample_id in processed_sample_ids:
            print(f"Skipping processed sample: {sample_id}")
            continue

        sample_dir = os.path.join(output_dir, f"sample_{sanitize_filename(sample_id)}")
        ensure_dir(sample_dir)
        qa_result_path = os.path.join(sample_dir, "qa_result.json")
        qa_result_jsonl_path = os.path.join(sample_dir, "qa_result.jsonl")
        token_stat_path = os.path.join(sample_dir, "token_stat.json")

        agent = SimpleMemAgent(
            model=model,
            backend=backend,
            retrieve_k=retrieve_k,
            api_key=api_key,
            base_url=base_url,
            embedding_model_name=embedding_model_name,
        )
        sample_stage_name = f"Sample {sample_id}"

        print(f"\nProcessing sample {sample_idx + 1}/{len(samples)} (sample_id={sample_id})")
        print("=" * 50)

        with tracker.stage(sample_stage_name):
            with tracker.stage("Memory Ingestion"):
                dialog_idx = 0
                for _, turns in sample.conversation.sessions.items():
                    for turn in turns.turns:
                        conversation_text = f"Speaker {turn.speaker} says : {turn.text}"
                        with tracker.stage(f"Dialog {dialog_idx}"):
                            agent.add_memory(conversation_text, time=turns.date_time)
                        dialog_idx += 1

            qa_results = []
            for qa_idx, qa in enumerate(sample.qa):
                total_questions += 1
                reference_answer = qa.final_answer if qa.final_answer is not None else qa.answer
                reference_answer = reference_answer or ""

                with tracker.stage(f"Retrieval QA {qa_idx}"):
                    answer, retrieved_memories = agent.answer_question(
                        qa.question,
                        qa.category,
                        reference_answer or "",
                        return_retrieved=True,
                    )

                qa_result = {
                    "qa_idx": qa_idx,
                    "question": qa.question,
                    "answer": reference_answer,
                    "category": qa.category,
                    "response": answer,
                    "retrieved": retrieved_memories,
                }
                qa_results.append(qa_result)

                with open(qa_result_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(qa_result, ensure_ascii=False) + "\n")

                print(f"\nQuestion {total_questions}: {qa.question}")
                print(f"Answer: {answer}")
                print(f"Category: {qa.category}")
                print("-" * 30)

        write_json_file(qa_result_path, qa_results)
        write_sample_token_tree(tracker, sample_stage_name, token_stat_path)

        results.append(
            {
                "sample_id": sample_id,
                "qa": qa_results,
            }
        )
        processed_sample_ids.add(sample_id)

        write_json_file(output_path, results)
        rebuild_aggregate_token_file(output_dir, token_file)
        print(f"Sample {sample_id} results saved to {sample_dir}")

    write_json_file(output_path, results)
    rebuild_aggregate_token_file(output_dir, token_file)
    print(f"\nCompleted! Total questions answered in this run: {total_questions}")
    print(f"Aggregated results saved to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AMEM on a LOCOMO-style dataset.")
    parser.add_argument("--dataset_path", "--dataset", dest="dataset_path", type=str, required=True)
    parser.add_argument("--llm_model", "--model", dest="llm_model", type=str, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--modelname", type=str, default=None)
    parser.add_argument("--output_path", "--output", dest="output_path", type=str, default=None)
    parser.add_argument("--token_file", type=str, default=None)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--backend", type=str, default="openai")
    parser.add_argument("--retrieve_k", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--llm_api_key", type=str, default=DEFAULT_LLM_API_KEY)
    parser.add_argument("--llm_base_url", type=str, default=DEFAULT_LLM_BASE_URL)
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL_NAME,
    )
    args = parser.parse_args()

    simple_qa_session(
        dataset_path=args.dataset_path,
        model=args.llm_model,
        modelname=args.modelname,
        output_path=args.output_path,
        ratio=args.ratio,
        backend=args.backend,
        retrieve_k=args.retrieve_k,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        embedding_model_name=args.embedding_model_name,
        token_file=args.token_file,
    )
