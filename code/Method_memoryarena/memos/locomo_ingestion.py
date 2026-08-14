import contextlib
import os
import shutil
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd

from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

from memos.configs.mem_cube import GeneralMemCubeConfig
from memos.configs.mem_os import MOSConfig
from memos.mem_cube.general import GeneralMemCube
from memos.mem_os.main import MOS

from .configuration import (
    build_mem_cube_config,
    build_mos_config,
    ensure_dir,
    ensure_parent_dir,
    get_storage_path,
)
from .token_tracker import TokenTracker
from ..dataset_hygiene import format_turn_text, natural_session_keys


def get_client(user_id: str, runtime_config):
    mos_config_data = build_mos_config(
        runtime_config,
        top_k=runtime_config.get("ingestion_top_k", 20),
    )
    mos_config = MOSConfig(**mos_config_data)
    mos = MOS(mos_config)
    mos.create_user(user_id=user_id)

    mem_cube_config_data = build_mem_cube_config(runtime_config, user_id)
    mem_cube_config = GeneralMemCubeConfig.model_validate(mem_cube_config_data)
    mem_cube = GeneralMemCube(mem_cube_config)

    storage_path = get_storage_path(runtime_config, user_id)
    ensure_dir(runtime_config["storage_dir"])
    try:
        mem_cube.dump(storage_path)
    except Exception as e:
        print(f"dumping memory cube: {e!s} already exists, will use it")

    mos.register_mem_cube(
        mem_cube_name_or_path=storage_path,
        mem_cube_id=user_id,
        user_id=user_id,
    )

    return mos


def ingest_session(client, session, metadata, revised_client, tracker: TokenTracker | None):
    session_date = metadata["session_date"]
    date_format = "%I:%M %p on %d %B, %Y UTC"
    date_string = datetime.strptime(session_date, date_format).replace(tzinfo=timezone.utc)
    iso_date = date_string.isoformat()
    conv_idx = metadata["conv_idx"]
    conv_id = "locomo_exp_user_" + str(conv_idx)
    print(f"Processing conv {conv_id}, session {metadata['session_key']}")
    start_time = time.time()

    messages = []
    messages_reverse = []

    for chat in tqdm(session, desc=f"{metadata['session_key']}"):
        data = f"{chat.get('speaker', 'Unknown')}: {format_turn_text(chat)}"

        if chat.get("speaker") == metadata["speaker_a"]:
            messages.append({"role": "user", "content": data, "chat_time": iso_date})
            messages_reverse.append(
                {"role": "assistant", "content": data, "chat_time": iso_date}
            )
        elif chat.get("speaker") == metadata["speaker_b"]:
            messages.append({"role": "assistant", "content": data, "chat_time": iso_date})
            messages_reverse.append({"role": "user", "content": data, "chat_time": iso_date})
        else:
            raise ValueError(
                f"Unknown speaker {chat.get('speaker')} in session {metadata['session_key']}"
            )

        print({"context": data, "conv_id": conv_id, "created_at": iso_date})

    speaker_a_user_id = conv_id + "_speaker_a"
    speaker_b_user_id = conv_id + "_speaker_b"
    session_key = metadata["session_key"]
    sample_stage = tracker.stage(f"Sample {conv_idx}") if tracker else contextlib.nullcontext()
    session_stage = (
        tracker.stage(f"Session {session_key}") if tracker else contextlib.nullcontext()
    )
    with sample_stage:
        with session_stage:
            client.add(
                messages=messages,
                user_id=speaker_a_user_id,
            )
            revised_client.add(
                messages=messages_reverse,
                user_id=speaker_b_user_id,
            )
    print(f"Added messages for {speaker_a_user_id} and {speaker_b_user_id} successfully.")

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    return elapsed_time


def process_user(conv_idx, locomo_df, runtime_config, tracker):
    try:
        conversation = locomo_df["conversation"].iloc[conv_idx]
        start_time = time.time()
        total_session_time = 0
        valid_sessions = 0

        conv_id = "locomo_exp_user_" + str(conv_idx)
        speaker_a_user_id = conv_id + "_speaker_a"
        speaker_b_user_id = conv_id + "_speaker_b"
        client = get_client(speaker_a_user_id, runtime_config)
        revised_client = get_client(speaker_b_user_id, runtime_config)

        sessions_to_process = []
        for session_key in natural_session_keys(conversation):
            session = conversation[session_key]

            metadata = {
                "session_date": str(conversation.get(f"{session_key}_date_time") or "") + " UTC",
                "speaker_a": conversation.get("speaker_a"),
                "speaker_b": conversation.get("speaker_b"),
                "speaker_a_user_id": f"{conversation.get('speaker_a')}_{conv_idx}",
                "speaker_b_user_id": f"{conversation.get('speaker_b')}_{conv_idx}",
                "conv_idx": conv_idx,
                "session_key": session_key,
            }
            sessions_to_process.append((session, metadata))
            valid_sessions += 1

        print(f"Processing {valid_sessions} sessions for user {conv_idx} sequentially")
        for session, metadata in sessions_to_process:
            session_key = metadata["session_key"]
            session_time = ingest_session(client, session, metadata, revised_client, tracker)
            total_session_time += session_time
            print(f"User {conv_idx}, {session_key} processed in {session_time} seconds")

        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        print(f"User {conv_idx} processed successfully in {elapsed_time} seconds")
        return elapsed_time

    except Exception as e:
        return f"Error processing user {conv_idx}: {e!s}"


def clear_graph_databases(runtime_config, num_users):
    graph_config = runtime_config["mem_cube_config_template"]["text_mem"]["config"][
        "graph_db"
    ]["config"]
    uri = graph_config.get("uri")
    user = graph_config.get("user")
    password = graph_config.get("password")
    if not all((uri, user, password)):
        raise ValueError("Neo4j URI, user, and password are required for database cleanup")

    database_names = []
    for conv_idx in range(num_users):
        for speaker in ("a", "b"):
            user_id = f"locomo_exp_user_{conv_idx}_speaker_{speaker}"
            mem_cube_config = build_mem_cube_config(runtime_config, user_id)
            db_name = mem_cube_config["text_mem"]["config"]["graph_db"]["config"][
                "db_name"
            ]
            if db_name in {"neo4j", "system"}:
                raise ValueError(f"Refusing to drop protected Neo4j database: {db_name}")
            database_names.append(db_name)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database="system") as session:
            for db_name in database_names:
                escaped_name = db_name.replace("`", "``")
                session.run(f"DROP DATABASE `{escaped_name}` IF EXISTS").consume()
                deadline = time.monotonic() + 60
                while True:
                    database = session.run(
                        """
                        SHOW DATABASES YIELD name
                        WHERE name = $name
                        RETURN name
                        """,
                        name=db_name,
                    ).single()
                    if database is None:
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out dropping Neo4j database: {db_name}")
                    time.sleep(0.5)
                print(f"Dropped stale Neo4j database if present: {db_name}")
    finally:
        driver.close()


def prepare_ingestion(runtime_config, num_users):
    if runtime_config.get("clear_existing_index") and os.path.isdir(
        runtime_config["storage_dir"]
    ):
        shutil.rmtree(runtime_config["storage_dir"])
        print(f"Removed stale Memos storage: {runtime_config['storage_dir']}")

    if runtime_config.get("clear_previous_results"):
        for top_k in runtime_config["retrieve_top_ks"]:
            result_dir = os.path.join(runtime_config["result_dir"], f"top_k_{top_k}")
            if os.path.isdir(result_dir):
                shutil.rmtree(result_dir)
                print(f"Removed stale top-k results: {result_dir}")

    if runtime_config.get("clear_graph_db"):
        clear_graph_databases(runtime_config, num_users)


def ingestion(runtime_config):
    tracker = None
    if runtime_config.get("track_tokens"):
        token_file = runtime_config["token_file"]
        ensure_parent_dir(token_file)
        tracker = TokenTracker(output_file=token_file)
        tracker.patch_llm_api()

    load_dotenv()
    locomo_df = pd.read_json(runtime_config["dataset_path"])
    num_users = locomo_df.shape[0]
    prepare_ingestion(runtime_config, num_users)

    start_time = time.time()
    total_time = 0
    num_workers = max(1, int(runtime_config.get("num_workers", 1)))

    print(f"Starting processing for {num_users} users with {num_workers} worker(s)...")

    if num_workers == 1:
        results = [
            process_user(user_id, locomo_df, runtime_config, tracker)
            for user_id in range(num_users)
        ]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_user, user_id, locomo_df, runtime_config, tracker): user_id
                for user_id in range(num_users)
            }
            for future in as_completed(futures):
                user_id = futures[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(f"Error processing user {user_id}: {e!s}")

    errors = []
    for result in results:
        if isinstance(result, float):
            total_time += result
        else:
            errors.append(str(result))
            print(result)

    if errors:
        raise RuntimeError(
            f"Memos indexing failed for {len(errors)} user(s): " + "; ".join(errors)
        )

    if num_users > 0:
        average_time = total_time / num_users
        minutes = int(average_time // 60)
        seconds = int(average_time % 60)
        average_time_formatted = f"{minutes} minutes and {seconds} seconds"
        print(
            f"Memos framework processed {num_users} users in average of {average_time_formatted} per user."
        )

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    elapsed_time = f"{minutes} minutes and {seconds} seconds"
    print(f"Total processing time: {elapsed_time}.")


if __name__ == "__main__":
    import argparse

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
    args = parser.parse_args()

    runtime_config = build_runtime_config(
        {
            "version": args.version,
            "num_workers": args.workers,
        }
    )
    ingestion(runtime_config)
