from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import requests


LOCAL_HOSTS = {"127.0.0.1", "0.0.0.0", "localhost", "::1"}
PORT_FLAG_RE = re.compile(r"(?:^|\s)--port(?:=|\s+)(\d+)(?:\s|$)")


def _default_printer(message: str) -> None:
    print(message, flush=True)


def is_local_base_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").strip().lower()
    return host in LOCAL_HOSTS


def extract_port_from_base_url(base_url: str) -> int:
    parsed = urlparse(base_url)
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "http":
        return 80
    if parsed.scheme == "https":
        return 443
    raise ValueError(f"Unable to infer port from upstream_env_server_base={base_url!r}")


def render_port_template(value: Optional[str], port: int) -> Optional[str]:
    if value is None:
        return None
    return str(value).replace("{port}", str(port))


def save_env_id(env_id: int, cache_file: str | Path) -> None:
    path = Path(cache_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"env_id": env_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)


def load_env_id(cache_file: str | Path) -> Optional[int]:
    path = Path(cache_file)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        env_id = payload.get("env_id")
        return int(env_id) if env_id is not None else None
    except Exception:
        return None


def _iter_repo_python_cache_paths(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for cache_dir in repo_root.rglob("__pycache__"):
        if cache_dir.is_dir():
            paths.append(cache_dir)
    for suffix in ("*.pyc", "*.pyo"):
        paths.extend(path for path in repo_root.rglob(suffix) if path.is_file())
    return paths


def clear_python_caches(repo_root: Path, printer: Callable[[str], None] = _default_printer) -> int:
    removed = 0
    for path in _iter_repo_python_cache_paths(repo_root):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    printer(f"[shopping-bootstrap] Cleared {removed} Python cache paths under {repo_root}")
    return removed


def clear_env_id_caches(
    cache_file: Optional[str | Path],
    port: int,
    repo_root: Path,
    printer: Callable[[str], None] = _default_printer,
) -> list[str]:
    removed: list[str] = []
    candidates: set[Path] = set()
    if cache_file:
        candidates.add(Path(cache_file))

    legacy_patterns = (
        f".env_id_cache_{port}.json",
        f".env_id_cache_{port}_*.json",
    )
    for root in {repo_root, Path.cwd()}:
        for pattern in legacy_patterns:
            candidates.update(root.glob(pattern))

    for path in sorted(candidates):
        try:
            if path.exists():
                path.unlink()
                removed.append(str(path.resolve()))
        except FileNotFoundError:
            continue

    if removed:
        printer("[shopping-bootstrap] Cleared env-id cache files:")
        for path in removed:
            printer(f"  - {path}")
    else:
        printer("[shopping-bootstrap] No env-id cache files needed clearing")
    return removed


def _list_upstream_processes(port: int, launch_module: str) -> list[tuple[int, str]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        check=True,
    )
    matches: list[tuple[int, str]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or launch_module not in line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        port_match = PORT_FLAG_RE.search(command)
        if port_match and int(port_match.group(1)) == port:
            matches.append((int(pid_text), command))
    return matches


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop_launch_lite_processes(
    port: int,
    launch_module: str = "env.env_systems.web_shopping_env.runtime.service.launch_lite",
    printer: Callable[[str], None] = _default_printer,
    grace_seconds: float = 15.0,
) -> list[int]:
    processes = _list_upstream_processes(port, launch_module)
    if not processes:
        printer(
            f"[shopping-bootstrap] No existing upstream process found for "
            f"{launch_module} on port {port}"
        )
        return []

    pids = [pid for pid, _ in processes]
    printer(
        f"[shopping-bootstrap] Stopping existing upstream processes for "
        f"{launch_module} on port {port}: {pids}"
    )
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue

    deadline = time.time() + grace_seconds
    while time.time() < deadline and any(_pid_exists(pid) for pid in pids):
        time.sleep(0.5)

    stubborn = [pid for pid in pids if _pid_exists(pid)]
    for pid in stubborn:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
    if stubborn:
        printer(f"[shopping-bootstrap] Force killed upstream processes: {stubborn}")
    return pids


def is_service_reachable(base_url: str, timeout: float = 2.0) -> bool:
    try:
        response = requests.get(base_url.rstrip("/") + "/", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def wait_for_service_ready(
    base_url: str,
    timeout_seconds: float,
    poll_interval: float = 2.0,
    process: Optional[subprocess.Popen[bytes]] = None,
    log_file: Optional[Path] = None,
    printer: Callable[[str], None] = _default_printer,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if is_service_reachable(base_url):
            printer(f"[shopping-bootstrap] Upstream service responded at {base_url}")
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(_build_start_failure_message(process.returncode, log_file))
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for upstream service at {base_url}")


def _build_start_failure_message(returncode: Optional[int], log_file: Optional[Path]) -> str:
    message = f"Upstream launch_lite exited early with return code {returncode}"
    if log_file and log_file.exists():
        try:
            tail = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        except Exception:
            tail = []
        if tail:
            message += "\nLast log lines:\n" + "\n".join(tail)
        else:
            message += f"\nLog file: {log_file}"
    return message


def _prepend_env_path(env: dict[str, str], path: Path) -> None:
    current = env.get("PATH", "")
    path_str = str(path)
    parts = [part for part in current.split(os.pathsep) if part]
    if path_str not in parts:
        env["PATH"] = path_str if not current else path_str + os.pathsep + current


def _infer_prefix_from_python_executable(python_executable: str) -> Optional[Path]:
    python_path = Path(python_executable).resolve()
    if python_path.parent.name != "bin":
        return None
    return python_path.parent.parent


def _infer_jvm_path(prefix: Path) -> Optional[Path]:
    candidates = (
        prefix / "lib" / "jvm" / "lib" / "server" / "libjvm.so",
        prefix / "lib" / "server" / "libjvm.so",
        prefix / "jre" / "lib" / "amd64" / "server" / "libjvm.so",
        prefix / "jre" / "lib" / "amd64" / "default" / "libjvm.so",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def warm_upstream_env(
    base_url: str,
    cache_file: Optional[str | Path],
    timeout_seconds: float,
    reuse_env: bool,
    printer: Callable[[str], None] = _default_printer,
) -> Optional[int]:
    cache_path = Path(cache_file) if cache_file else None
    cached_env_id = load_env_id(cache_path) if reuse_env and cache_path else None
    if cached_env_id is not None:
        try:
            response = requests.get(
                base_url.rstrip("/") + f"/observation?env_idx={cached_env_id}",
                timeout=min(timeout_seconds, 30.0),
            )
            if response.status_code == 200:
                printer(f"[shopping-bootstrap] Reusing cached upstream env_id={cached_env_id}")
                return cached_env_id
        except Exception:
            pass

    printer("[shopping-bootstrap] Priming upstream WebShop env via /create")
    response = requests.post(base_url.rstrip("/") + "/create", timeout=timeout_seconds)
    response.raise_for_status()
    env_id = int(response.json())
    verify = requests.get(
        base_url.rstrip("/") + f"/observation?env_idx={env_id}",
        timeout=min(timeout_seconds, 30.0),
    )
    verify.raise_for_status()
    if reuse_env and cache_path is not None:
        save_env_id(env_id, cache_path)
        printer(f"[shopping-bootstrap] Saved warmed upstream env_id={env_id} -> {cache_path}")
    else:
        printer(
            "[shopping-bootstrap] Warmed upstream WebShop env via /create "
            f"(env_id={env_id}, reuse_env={reuse_env})"
        )
    return env_id


def _launch_process(
    repo_root: Path,
    python_executable: str,
    port: int,
    launch_module: str,
    limit_goals: int,
    log_file: Optional[Path],
    env_overrides: Optional[dict[str, str]] = None,
    printer: Callable[[str], None] = _default_printer,
) -> subprocess.Popen[bytes]:
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_file, "ab")
    else:
        log_handle = subprocess.DEVNULL

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    inferred_prefix = _infer_prefix_from_python_executable(python_executable)
    if inferred_prefix is not None:
        inferred_bin = inferred_prefix / "bin"
        if inferred_bin.exists():
            _prepend_env_path(env, inferred_bin)
        inferred_javac = inferred_bin / "javac"
        if not env.get("JAVA_HOME") and inferred_javac.exists():
            env["JAVA_HOME"] = str(inferred_prefix)
            printer(
                "[shopping-bootstrap] Inferred JAVA_HOME from upstream python env: "
                f"{inferred_prefix}"
            )
        inferred_jvm = _infer_jvm_path(inferred_prefix)
        if inferred_jvm is not None and not env.get("JVM_PATH"):
            env["JVM_PATH"] = str(inferred_jvm)
            printer(
                "[shopping-bootstrap] Inferred JVM_PATH from upstream python env: "
                f"{inferred_jvm}"
            )
    if env_overrides:
        env.update({key: str(value) for key, value in env_overrides.items() if value is not None})
    command = [
        python_executable,
        "-m",
        launch_module,
        "--port",
        str(port),
        "--limit_goals",
        str(limit_goals),
    ]
    printer(f"[shopping-bootstrap] Launching upstream service: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    finally:
        if log_file is not None and log_handle is not subprocess.DEVNULL:
            log_handle.close()
    return process


@dataclass
class BootstrapResult:
    base_url: str
    port: int
    launched: bool
    restarted: bool
    warmed_env_id: Optional[int]
    cache_file: Optional[str]
    log_file: Optional[str]


def ensure_upstream_webshop_service(
    *,
    base_url: str,
    repo_root: Path,
    reuse_env: bool,
    cache_file: Optional[str | Path],
    restart: bool = False,
    python_executable: Optional[str] = None,
    launch_module: str = "env.env_systems.web_shopping_env.runtime.service.launch_lite",
    limit_goals: int = -1,
    ready_timeout_seconds: float = 600.0,
    clear_env_id_cache_on_restart: bool = True,
    clear_python_cache_on_restart: bool = False,
    log_file: Optional[str | Path] = None,
    env_overrides: Optional[dict[str, str]] = None,
    printer: Callable[[str], None] = _default_printer,
) -> BootstrapResult:
    if not is_local_base_url(base_url):
        raise ValueError(
            "bootstrap_upstream_env only supports local upstream_env_server_base "
            f"(got {base_url})"
        )

    port = extract_port_from_base_url(base_url)
    cache_path = Path(cache_file).resolve() if cache_file else None
    log_path = Path(log_file).resolve() if log_file else None
    python_bin = python_executable or sys.executable
    launched = False
    restarted = False

    if restart or not is_service_reachable(base_url):
        restarted = restart
        if clear_env_id_cache_on_restart:
            clear_env_id_caches(cache_path, port, repo_root, printer=printer)
        if clear_python_cache_on_restart:
            clear_python_caches(repo_root, printer=printer)
        stop_launch_lite_processes(port, launch_module=launch_module, printer=printer)
        process = _launch_process(
            repo_root=repo_root,
            python_executable=python_bin,
            port=port,
            launch_module=launch_module,
            limit_goals=limit_goals,
            log_file=log_path,
            env_overrides=env_overrides,
            printer=printer,
        )
        launched = True
        wait_for_service_ready(
            base_url=base_url,
            timeout_seconds=ready_timeout_seconds,
            process=process,
            log_file=log_path,
            printer=printer,
        )
    else:
        printer(f"[shopping-bootstrap] Reusing already-running upstream service at {base_url}")

    warmed_env_id = warm_upstream_env(
        base_url=base_url,
        cache_file=cache_path,
        timeout_seconds=ready_timeout_seconds,
        reuse_env=reuse_env,
        printer=printer,
    )

    return BootstrapResult(
        base_url=base_url,
        port=port,
        launched=launched,
        restarted=restarted,
        warmed_env_id=warmed_env_id,
        cache_file=str(cache_path) if cache_path else None,
        log_file=str(log_path) if log_path else None,
    )
