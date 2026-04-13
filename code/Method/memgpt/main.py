import os
import signal
import socket
import subprocess
import time
from urllib.parse import urlsplit

import requests

from .letta_workflow import (
    DEFAULT_EMBEDDING_API_KEY,
    DEFAULT_EMBEDDING_BASE_URL,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_ENDPOINT_TYPE,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_LETTA_BASE_URL,
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_RETRIEVE_K,
    run_memgpt_workflow,
)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_PATH = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../Result/LOCOMO/memgpt/default/result.json")
)


def resolve_path(path_value, base_dir):
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def normalize_url(url):
    return (url or "").rstrip("/")


def build_default_letta_server_command(letta_base_url):
    parsed = urlsplit(letta_base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f"letta server --host {host} --port {port}"


def parse_pg_host_port(pg_uri):
    parsed = urlsplit(pg_uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    return host, port


def is_postgres_reachable(pg_uri, timeout=3):
    try:
        host, port = parse_pg_host_port(pg_uri)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def is_letta_healthy(letta_base_url, timeout=3):
    base_url = normalize_url(letta_base_url)
    health_paths = ["/v1/health", "/v1/health/", "/latest/health/"]
    for path in health_paths:
        try:
            response = requests.get(f"{base_url}{path}", timeout=timeout)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            continue
    return False


def build_runtime_env(letta_pg_uri, llm_api_key, llm_base_url, embedding_api_key, embedding_base_url):
    env = os.environ.copy()

    if letta_pg_uri:
        env["LETTA_PG_URI"] = letta_pg_uri

    preferred_api_key = embedding_api_key or llm_api_key
    preferred_base_url = embedding_base_url or llm_base_url

    if preferred_api_key:
        env["OPENAI_API_KEY"] = preferred_api_key
        env["LETTA_OPENAI_API_KEY"] = preferred_api_key

    if preferred_base_url:
        env["OPENAI_BASE_URL"] = preferred_base_url
        env["LETTA_OPENAI_API_BASE"] = preferred_base_url

    env.setdefault("LETTA_SERVER_SECURE", "false")
    return env


def run_shell_command(command, env, cwd, description):
    if not command:
        return

    completed = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{description} failed with exit code {completed.returncode}.\n"
            f"Command: {command}\n"
            f"Stdout:\n{completed.stdout}\n"
            f"Stderr:\n{completed.stderr}"
        )


def start_letta_server(command, env, cwd, log_path):
    log_file = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=cwd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return process, log_file


def wait_for_letta_server(letta_base_url, timeout_seconds, process, log_path):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if is_letta_healthy(letta_base_url, timeout=2):
            return
        if process.poll() is not None:
            raise RuntimeError(
                f"Letta server exited early with code {process.returncode}. "
                f"Check the log at: {log_path}"
            )
        time.sleep(1)

    raise TimeoutError(
        f"Timed out waiting for Letta server at {letta_base_url}. "
        f"Check the startup log at: {log_path}"
    )


def stop_started_server(process, log_file):
    if process is None:
        return

    try:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass
    finally:
        if log_file is not None:
            log_file.close()


def run_memgpt(
    dataset_path,
    output_path=None,
    token_file=None,
    llm_model=DEFAULT_LLM_MODEL,
    llm_api_key=DEFAULT_LLM_API_KEY,
    llm_base_url=DEFAULT_LLM_BASE_URL,
    embedding_model_name=DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_api_key=DEFAULT_EMBEDDING_API_KEY,
    embedding_base_url=DEFAULT_EMBEDDING_BASE_URL,
    embedding_endpoint_type=DEFAULT_EMBEDDING_ENDPOINT_TYPE,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    letta_base_url=DEFAULT_LETTA_BASE_URL,
    letta_pg_uri=None,
    letta_server_command=None,
    letta_init_command=None,
    letta_startup_timeout=90,
    retrieve_k=DEFAULT_RETRIEVE_K,
    start_idx=0,
    end_idx=None,
    config_path=None,
):
    base_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else os.getcwd()

    dataset_path = resolve_path(dataset_path, base_dir)
    if not dataset_path:
        raise ValueError("MemGPT requires 'dataset_path' in the config or CLI arguments.")

    output_path = resolve_path(output_path, base_dir) or DEFAULT_OUTPUT_PATH
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    token_file = resolve_path(token_file, base_dir) or os.path.join(output_dir, "token_tracker.json")
    letta_base_url = normalize_url(letta_base_url or DEFAULT_LETTA_BASE_URL)
    letta_server_command = letta_server_command or build_default_letta_server_command(letta_base_url)
    letta_startup_timeout = int(letta_startup_timeout or 90)

    runtime_env = build_runtime_env(
        letta_pg_uri=letta_pg_uri or os.environ.get("LETTA_PG_URI"),
        llm_api_key=llm_api_key or DEFAULT_LLM_API_KEY,
        llm_base_url=llm_base_url or DEFAULT_LLM_BASE_URL,
        embedding_api_key=embedding_api_key or DEFAULT_EMBEDDING_API_KEY,
        embedding_base_url=embedding_base_url or DEFAULT_EMBEDDING_BASE_URL,
    )

    started_process = None
    started_log = None
    started_server_here = False
    log_path = os.path.join(output_dir, "letta_server.log")

    try:
        if not is_letta_healthy(letta_base_url):
            effective_pg_uri = runtime_env.get("LETTA_PG_URI")
            if not effective_pg_uri:
                raise ValueError(
                    "MemGPT could not find a running Letta server and no `letta_pg_uri` was provided."
                )
            if not is_postgres_reachable(effective_pg_uri):
                raise ConnectionError(
                    f"PostgreSQL is not reachable for Letta at {effective_pg_uri}. "
                    "Please start PostgreSQL and ensure pgvector is enabled."
                )

            run_shell_command(
                letta_init_command,
                env=runtime_env,
                cwd=base_dir,
                description="Letta initialization command",
            )

            started_process, started_log = start_letta_server(
                command=letta_server_command,
                env=runtime_env,
                cwd=base_dir,
                log_path=log_path,
            )
            started_server_here = True
            wait_for_letta_server(
                letta_base_url=letta_base_url,
                timeout_seconds=letta_startup_timeout,
                process=started_process,
                log_path=log_path,
            )

        return run_memgpt_workflow(
            dataset_path=dataset_path,
            output_path=output_path,
            token_file=token_file,
            llm_model=llm_model or DEFAULT_LLM_MODEL,
            llm_api_key=llm_api_key or DEFAULT_LLM_API_KEY,
            llm_base_url=llm_base_url or DEFAULT_LLM_BASE_URL,
            embedding_model_name=embedding_model_name or DEFAULT_EMBEDDING_MODEL_NAME,
            embedding_api_key=embedding_api_key or DEFAULT_EMBEDDING_API_KEY,
            embedding_base_url=embedding_base_url or DEFAULT_EMBEDDING_BASE_URL,
            embedding_endpoint_type=embedding_endpoint_type or DEFAULT_EMBEDDING_ENDPOINT_TYPE,
            embedding_dim=embedding_dim or DEFAULT_EMBEDDING_DIM,
            letta_base_url=letta_base_url,
            retrieve_k=retrieve_k if retrieve_k is not None else DEFAULT_RETRIEVE_K,
            start_idx=start_idx if start_idx is not None else 0,
            end_idx=end_idx,
        )
    finally:
        if started_server_here:
            stop_started_server(started_process, started_log)
