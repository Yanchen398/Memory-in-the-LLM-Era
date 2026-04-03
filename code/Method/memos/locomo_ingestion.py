import time

from datetime import datetime, timezone

import pandas as pd

from dotenv import load_dotenv
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


def ingest_session(client, session, metadata, revised_client, tracker: TokenTracker):
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
        data = chat.get("speaker") + ": " + chat.get("text")

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
    with tracker.stage(f"Sample {conv_idx}"):
        with tracker.stage(f"Session {session_key}"):
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
        max_session_count = 35
        start_time = time.time()
        total_session_time = 0
        valid_sessions = 0

        conv_id = "locomo_exp_user_" + str(conv_idx)
        speaker_a_user_id = conv_id + "_speaker_a"
        speaker_b_user_id = conv_id + "_speaker_b"
        client = get_client(speaker_a_user_id, runtime_config)
        revised_client = get_client(speaker_b_user_id, runtime_config)

        sessions_to_process = []
        for session_idx in range(max_session_count):
            session_key = f"session_{session_idx}"
            session = conversation.get(session_key)
            if session is None:
                continue

            metadata = {
                "session_date": conversation.get(f"session_{session_idx}_date_time") + " UTC",
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
            try:
                session_time = ingest_session(client, session, metadata, revised_client, tracker)
                total_session_time += session_time
                print(f"User {conv_idx}, {session_key} processed in {session_time} seconds")
            except Exception as e:
                print(f"Error processing user {conv_idx}, session {session_key}: {e!s}")

        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        print(f"User {conv_idx} processed successfully in {elapsed_time} seconds")
        return elapsed_time

    except Exception as e:
        return f"Error processing user {conv_idx}: {e!s}"


def ingestion(runtime_config):
    token_file = runtime_config["token_file"]
    ensure_parent_dir(token_file)
    tracker = TokenTracker(output_file=token_file)
    tracker.patch_llm_api()
    load_dotenv()
    locomo_df = pd.read_json(runtime_config["dataset_path"])

    num_users = locomo_df.shape[0]
    start_time = time.time()
    total_time = 0

    print(f"Starting processing for {num_users} users in serial mode...")

    for user_id in range(num_users):
        try:
            result = process_user(user_id, locomo_df, runtime_config, tracker)
            if isinstance(result, float):
                total_time += result
            else:
                print(result)
        except Exception as e:
            print(f"Error processing user {user_id}: {e!s}")

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
