import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv
from openai import AsyncOpenAI

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EpisodeType

from .token_tracker import TokenTracker


DEFAULT_LLM_API_KEY = "empty"
DEFAULT_LLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_LLM_MODEL = "/home/docker/LLaMA-Factory/output/qwen2_5_lora_sft"
DEFAULT_EMBEDDING_API_KEY = "empty"
DEFAULT_EMBEDDING_BASE_URL = "http://localhost:7999/v1"
DEFAULT_EMBEDDING_MODEL = "/home/docker/Model/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_NEO4J_URI = "bolt://localhost:7687"
DEFAULT_NEO4J_USER = "neo4j"
DEFAULT_NEO4J_PASSWORD = "neo4jneo4j"
DEFAULT_ANSWER_TEMPERATURE = 0.0
DEFAULT_ANSWER_MAX_TOKENS = 200


logging.basicConfig(
    level=INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


ANSWER_SYSTEM_PROMPT = """You are a helpful expert assistant answering user questions from retrieved conversation memories.
Answer briefly and precisely using only the provided context. If the context is insufficient, abstain.

When interpreting memories, use the timestamp to determine when an event happened, not when someone talked about it.

Example:
Memory: (2023-03-15T16:33:00Z) I went to the vet yesterday.
Question: What day did I go to the vet?
Correct answer: March 15, 2023
"""


def ensure_parent_dir(path):
    if path:
        parent_dir = os.path.dirname(path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)


def parse_datetime_string(datetime_str):
    """Parse a LOCOMO datetime string like '1:56 pm on 8 May, 2023'."""
    pattern = r"(\d{1,2}):(\d{2})\s+(am|pm)\s+on\s+(\d{1,2})\s+(\w+),\s+(\d{4})"
    match = re.match(pattern, datetime_str)

    if not match:
        raise ValueError(f"Cannot parse datetime string: {datetime_str}")

    hour, minute, ampm, day, month_name, year = match.groups()

    hour = int(hour)
    if ampm.lower() == "pm" and hour != 12:
        hour += 12
    elif ampm.lower() == "am" and hour == 12:
        hour = 0

    month_map = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }

    month = month_map.get(month_name)
    if not month:
        raise ValueError(f"Unknown month: {month_name}")

    return datetime(int(year), month, int(day), hour, int(minute), tzinfo=timezone.utc)


def format_dialogue_episode(speaker, text, blip_caption=None):
    """Format a dialogue message into a Graphiti episode body."""
    episode_body = f"\n{speaker}: {text}"
    if blip_caption:
        episode_body += f" (image caption: {blip_caption})"
    return episode_body


def extract_retrieved_facts(results):
    retrieved_facts = {"Edges": [], "Nodes": [], "Episodes": [], "Communities": []}

    try:
        for edge in results.edges[:5]:
            fact = f"{edge.name}: {edge.fact}"
            time_info = []
            if hasattr(edge, "expired_at") and edge.expired_at:
                time_info.append(f" (Expired at: {edge.expired_at})")
            if hasattr(edge, "valid_at") and edge.valid_at:
                time_info.append(f" (Valid from: {edge.valid_at})")
            if hasattr(edge, "invalid_at") and edge.invalid_at:
                time_info.append(f" (Valid until: {edge.invalid_at})")
            if time_info:
                fact += "".join(time_info)
            retrieved_facts["Edges"].append(fact)
    except Exception:
        pass

    try:
        for node in results.nodes[:5]:
            retrieved_facts["Nodes"].append(f"{node.name}: {node.summary}")
    except Exception:
        pass

    try:
        for episode in results.episodes:
            retrieved_facts["Episodes"].append(
                f"{episode.source_description}: {episode.content}"
            )
    except Exception:
        pass

    try:
        for community in results.communities[:3]:
            retrieved_facts["Communities"].append(f"{community.name}: {community.summary}")
    except Exception:
        pass

    return retrieved_facts


def build_answer_context(retrieved):
    sections = []
    for section_name in ("Edges", "Nodes", "Episodes", "Communities"):
        values = retrieved.get(section_name, [])
        if values:
            section_text = "\n".join(values)
            sections.append(f"[{section_name}]\n{section_text}")
    return "\n\n".join(sections)


def sample_has_completed_responses(sample_result):
    qa_items = sample_result.get("qa", [])
    if not qa_items:
        return False
    return all(item.get("response") for item in qa_items)


async def generate_answer(
    question,
    retrieved,
    answer_client,
    answer_model,
    answer_temperature=DEFAULT_ANSWER_TEMPERATURE,
    answer_max_tokens=DEFAULT_ANSWER_MAX_TOKENS,
):
    context = build_answer_context(retrieved)
    if not context.strip():
        return "Insufficient context to answer."

    try:
        response = await answer_client.chat.completions.create(
            model=answer_model,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
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
            max_tokens=answer_max_tokens,
            temperature=answer_temperature,
        )
        return response.choices[0].message.content.strip() if response.choices[0].message.content else "No response generated"
    except Exception as e:
        print(f"Error generating answer for question '{question}': {e}")
        return "Error: Unable to generate answer"


async def run_zep(
    dataset_path,
    output_path,
    token_file=None,
    llm_model=DEFAULT_LLM_MODEL,
    llm_small_model=None,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL,
    embedding_api_key=DEFAULT_EMBEDDING_API_KEY,
    embedding_base_url=DEFAULT_EMBEDDING_BASE_URL,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    neo4j_uri=DEFAULT_NEO4J_URI,
    neo4j_user=DEFAULT_NEO4J_USER,
    neo4j_password=DEFAULT_NEO4J_PASSWORD,
    answer_model=None,
    answer_api_key=None,
    answer_base_url=None,
    answer_temperature=DEFAULT_ANSWER_TEMPERATURE,
    answer_max_tokens=DEFAULT_ANSWER_MAX_TOKENS,
):
    load_dotenv()

    llm_model = llm_model or DEFAULT_LLM_MODEL
    llm_small_model = llm_small_model or llm_model
    llm_api_key = llm_api_key or DEFAULT_LLM_API_KEY
    llm_base_url = llm_base_url or DEFAULT_LLM_BASE_URL
    embedding_model_name = embedding_model_name or DEFAULT_EMBEDDING_MODEL
    embedding_api_key = embedding_api_key or DEFAULT_EMBEDDING_API_KEY
    embedding_base_url = embedding_base_url or DEFAULT_EMBEDDING_BASE_URL
    embedding_dim = embedding_dim or DEFAULT_EMBEDDING_DIM
    neo4j_uri = neo4j_uri or DEFAULT_NEO4J_URI
    neo4j_user = neo4j_user or DEFAULT_NEO4J_USER
    neo4j_password = neo4j_password or DEFAULT_NEO4J_PASSWORD
    answer_model = answer_model or llm_model
    answer_api_key = answer_api_key or llm_api_key
    answer_base_url = answer_base_url or llm_base_url
    token_file = token_file or output_path.replace(".json", "_tokens.json")

    ensure_parent_dir(output_path)
    ensure_parent_dir(token_file)

    tracker = TokenTracker(output_file=token_file)

    llm_config = LLMConfig(
        api_key=llm_api_key,
        model=llm_model,
        small_model=llm_small_model,
        base_url=llm_base_url,
    )
    llm_client = OpenAIGenericClient(config=llm_config)
    answer_client = AsyncOpenAI(api_key=answer_api_key, base_url=answer_base_url)

    graphiti = None
    try:
        graphiti = Graphiti(
            neo4j_uri,
            neo4j_user,
            neo4j_password,
            llm_client=llm_client,
            embedder=OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    embedding_model=embedding_model_name,
                    api_key=embedding_api_key,
                    base_url=embedding_base_url,
                    embedding_dim=embedding_dim,
                )
            ),
            cross_encoder=OpenAIRerankerClient(config=llm_config),
        )

        with open(dataset_path, "r", encoding="utf-8") as f:
            locomo_samples = json.load(f)

        all_results = []
        processed_sample_ids = set()
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                if existing_results:
                    completed_results = []
                    for result in existing_results:
                        if sample_has_completed_responses(result):
                            completed_results.append(result)
                            processed_sample_ids.add(result["sample_id"])
                        else:
                            print(f"Reprocessing incomplete sample result: {result.get('sample_id')}")
                    all_results = completed_results
                    print(f"Found existing results file with {len(completed_results)} completed samples")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load existing results from {output_path}: {e}")

        for sample_idx, locomo_data in enumerate(locomo_samples):
            namespace = locomo_data.get("sample_id", f"sample_{sample_idx}")
            if namespace in processed_sample_ids:
                print(
                    f"Skipping sample {sample_idx + 1}/{len(locomo_samples)} "
                    f"(ID: {namespace}) - already processed"
                )
                continue

            print(f"\n=== Processing sample {sample_idx + 1}/{len(locomo_samples)} (ID: {namespace}) ===")
            conversation = locomo_data["conversation"]
            session_keys = [
                key
                for key in conversation.keys()
                if key.startswith("session_") and not key.endswith("_date_time")
            ]

            with tracker.stage(f"Sample {namespace}"):
                for session_key in session_keys:
                    session_num = session_key.split("_")[1]
                    datetime_key = f"session_{session_num}_date_time"
                    if datetime_key not in conversation:
                        continue

                    datetime_str = conversation[datetime_key]
                    reference_time = parse_datetime_string(datetime_str)
                    session_dialogues = conversation[session_key]

                    with tracker.stage(f"Session {session_key}"):
                        for dialogue_idx, dialogue in enumerate(session_dialogues):
                            with tracker.stage(f"Dialog {dialogue_idx}"):
                                episode_body = format_dialogue_episode(
                                    dialogue["speaker"],
                                    dialogue["text"],
                                    dialogue.get("blip_caption"),
                                )
                                try:
                                    await graphiti.add_episode(
                                        name=f"Conversation Session {session_num} - Dialogue {dialogue_idx}",
                                        episode_body=episode_body,
                                        source=EpisodeType.message,
                                        source_description=(
                                            f"conversation between {conversation.get('speaker_a', 'Speaker A')} "
                                            f"and {conversation.get('speaker_b', 'Speaker B')} on {datetime_str}"
                                        ),
                                        reference_time=reference_time,
                                        group_id=namespace,
                                    )
                                except Exception as e:
                                    print(
                                        f"Error adding episode for sample {namespace}, "
                                        f"session {session_num}, dialogue {dialogue_idx}: {e}"
                                    )

            print(f"Finished processing sample {namespace}. Total sessions processed: {len(session_keys)}")
            print(f"Building communities for sample {namespace}...")
            await graphiti.build_communities(group_ids=[namespace])
            print(f"Communities built successfully for sample {namespace}.")

            qa_list = locomo_data.get("qa", [])
            print(f"\nSearching {len(qa_list)} questions for sample {namespace}:")

            sample_result = {"sample_id": namespace, "qa": []}
            for qa_idx, qa_item in enumerate(qa_list):
                if "question" not in qa_item:
                    continue

                question = qa_item["question"]
                expected_answer = qa_item.get("answer", "N/A")
                category = qa_item.get("category")

                print(f"\n--- Question {qa_idx + 1}/{len(qa_list)} ---")
                print(f"Question: {question}")
                print(f"Expected Answer: {expected_answer}")

                try:
                    results = await graphiti.search_(question, group_ids=[namespace])
                    retrieved_facts = extract_retrieved_facts(results)
                    response = await generate_answer(
                        question,
                        retrieved_facts,
                        answer_client,
                        answer_model,
                        answer_temperature=answer_temperature,
                        answer_max_tokens=answer_max_tokens,
                    )
                    qa_result = {
                        "question": question,
                        "answer": expected_answer,
                        "category": category,
                        "response": response,
                        "retrieved": retrieved_facts,
                    }
                except Exception as e:
                    print(f"Error during search/answer for question {qa_idx + 1}/{len(qa_list)} '{question}': {e}")
                    qa_result = {
                        "question": question,
                        "answer": expected_answer,
                        "category": category,
                        "response": "Error: Unable to generate answer",
                        "retrieved": {},
                    }

                sample_result["qa"].append(qa_result)

            all_results.append(sample_result)
            processed_sample_ids.add(namespace)

            try:
                ensure_parent_dir(output_path)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                print(
                    f"Results saved for sample {namespace}. "
                    f"Progress: {len(all_results)}/{len(locomo_samples)} samples completed"
                )
            except Exception as e:
                print(f"Warning: Could not save results for sample {namespace}: {e}")

        print(f"\n=== Finished processing all samples. Total samples in results: {len(all_results)} ===")
        ensure_parent_dir(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nFinal results saved to: {output_path}")
        total_questions = sum(len(sample["qa"]) for sample in all_results)
        print(f"Total questions processed: {total_questions}")
        return all_results

    finally:
        if graphiti is not None:
            await graphiti.close()
            print("\nConnection closed")


if __name__ == "__main__":
    asyncio.run(
        run_zep(
            dataset_path="/home/docker/IndepthMem/Dataset/LOCOMO/locomo10.json",
            output_path="/home/docker/IndepthMem/Result/LOCOMO/zep/result.json",
        )
    )
