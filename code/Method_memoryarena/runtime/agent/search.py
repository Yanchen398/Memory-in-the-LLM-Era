import json
import sys
from typing import Dict, List, Optional
from pathlib import Path
import openai

_CODE_ROOT = Path(__file__).resolve().parents[3]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from Method_memoryarena.client import MemoryClient
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess

# Standalone function for running a query with agent and memory
def run_query_with_agent_and_memory(
    query: str,
    memory_client,  # Now MemoryClient
    output_dir: Optional[Path] = None,
    script_path: str = "web_search_env/search_agent/openai_client.py",
    index_path: str = "web_search_env/embeddings/shard*.index",
    corpus_path: str = "web_search_env/data/corpus.jsonl",
    model: str = "gpt-5",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: Optional[str] = None,
    model_name: str = "text-embedding-ada-002",
    searcher_type: str = "openai",
    mcp_url: Optional[str] = None,
    mcp_name: str = "retrieval-mcp-server",
    gpu: Optional[str] = None,
    max_iterations: int = 9,
    max_tokens: int = 8192,
    retrieval_top_k: int = 10,
    snippet_max_tokens: int = 64,
    max_search_calls: int = 8,
    step_memory: bool = False,
) -> Dict:
    """
    Run a query with memory integration:
    1. Retrieve memory context using memory_client.wrap_user_prompt()
    2. Run agent with memory-augmented query
    3. Return complete trace + predicted answer
    Returns dict with: answer, trace (full result list), tool_call_counts, retrieved_docids, full_result
    """
    # Step 1: Retrieve memory context (no-op if memory_client is None, e.g. on env server without memory)
    if memory_client is None:
        query_with_memory = query
    else:
        query_with_memory = memory_client.wrap_user_prompt(query)
    memory_context = query_with_memory.replace(query, "").strip()

    print(f"[DEBUG] Query with memory (first 200 chars): {query_with_memory[:200]}")
    print(f"[DEBUG] Query with memory length: {len(query_with_memory)} chars")

    # Truncate query_with_memory if too long for OpenAI API
    MAX_OPENAI_TOKENS = 272000  # OpenAI API limit (use chars for safety)
    if len(query_with_memory) > MAX_OPENAI_TOKENS:
        print(f"[WARNING] Truncating query_with_memory from {len(query_with_memory)} to {MAX_OPENAI_TOKENS} characters for OpenAI safety.")
        query_with_memory = query_with_memory[:MAX_OPENAI_TOKENS]

    # Create output directory
    if output_dir is None:
        import tempfile
        output_dir = Path(tempfile.mkdtemp(prefix="tmp_query_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Write query to a temporary file to handle long queries
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tsv', encoding='utf-8') as f:
        escaped_query = query_with_memory.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        # Limit to avoid CSV field size limit (131072)
        if len(escaped_query) > 130000:
            escaped_query = escaped_query[:130000]
            print(f"[WARNING] Truncated query to 130000 characters to avoid CSV field limit")
        f.write(f"temp_query\t{escaped_query}\n")
        query_file = f.name

    print(f"[DEBUG] Written query to temp file: {query_file}")
    
    try:
        # Determine if using Anthropic client
        is_anthropic = mcp_url is not None or "anthropic" in script_path
        
        if is_anthropic:
            cmd = [
                sys.executable, "search_agent/anthropic_client.py",
                "--model", model,
                "--query", query_file,  # Pass file path
                "--output-dir", str(output_dir),
                "--max_tokens", str(max_tokens),
                "--mcp-url", mcp_url,
                "--mcp-name", mcp_name
            ]
        else:
            # Paths are relative to subprocess cwd (env_systems); use passed index_path so
            # e.g. web_search_env/embeddings/shard*.index finds env/env_systems/web_search_env/embeddings/
            id_map_path = (
                index_path.replace(".index", "_id_map.json")
                if index_path
                else "web_search_env/embeddings/shard*_id_map.json"
            )
            # LLM uses --model; searcher uses --openai-model (must be an embedding-capable model, e.g. text-embedding-3-small)
            cmd = [
                sys.executable, script_path,
                "--model", model,
                "--query", query_file,  # Pass file path
                "--output-dir", str(output_dir),
                "--searcher-type", searcher_type,
                "--max_tokens", str(max_tokens),
                "--max-iterations", str(max_iterations),
                "--max-search-calls", str(max_search_calls),
                "--k", str(retrieval_top_k),
                "--snippet-max-tokens", str(snippet_max_tokens)
            ]
            if searcher_type == "openai":
                cmd.extend([
                    "--index-path", index_path,
                    "--id-map-path", id_map_path,
                    "--corpus-path", corpus_path,
                    "--openai-model", model_name,
                ])
            elif searcher_type == "faiss":
                cmd.extend([
                    "--index-path", index_path,
                    "--model-name", model_name,
                    "--normalize",
                ])
            elif searcher_type == "bm25":
                cmd.extend(["--index-path", index_path])
            if provider:
                cmd.extend(["--provider", provider])
        
        # Set environment to use GPU
        env = os.environ.copy()
        if api_key is not None:
            env["OPENAI_API_KEY"] = api_key
        if base_url:
            env["OPENAI_BASE_URL"] = base_url
            env["OPENAI_API_BASE"] = base_url
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            print(f"[DEBUG] Setting CUDA_VISIBLE_DEVICES={gpu}")
        else:
            print(f"[DEBUG] Using inherited CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")

        # Expose memory identity to subprocess so it can do step-wise retrieval (only if step_memory=True)
        if step_memory and memory_client is not None:
            try:
                if hasattr(memory_client, "user_id"):
                    env["MEM_USER_ID"] = str(memory_client.user_id)
                if hasattr(memory_client, "memory_system_name"):
                    env["MEM_SYSTEM_NAME"] = str(memory_client.memory_system_name)
                if hasattr(memory_client, "base_url"):
                    env["MEM_BASE_URL"] = str(memory_client.base_url)
                print("[DEBUG] Step-wise memory enabled: MEM_USER_ID, MEM_SYSTEM_NAME, MEM_BASE_URL set for subprocess.")
            except Exception:
                # If anything goes wrong here, just skip step-wise memory; main run still works.
                pass
        
        actual_script = "search_agent/anthropic_client.py" if is_anthropic else script_path
        print(f"[DEBUG] Running agent: {sys.executable} {actual_script} --model {model}")
        print(f"[DEBUG] Command: {' '.join(cmd)}")
        print(f"[DEBUG] Output dir: {output_dir}")
        print(f"[DEBUG] Query file: {query_file}")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    finally:
        # Clean up temporary query file
        try:
            os.unlink(query_file)
        except:
            pass
    
    # Check for errors
    if result.returncode != 0:
        error_msg = f"Command failed with return code {result.returncode}\n"
        error_msg += f"STDOUT:\n{result.stdout}\n"
        error_msg += f"STDERR:\n{result.stderr}"
        raise RuntimeError(error_msg)
    
    # Show output for debugging
    if result.stderr:
        print(f"[DEBUG] STDERR: {result.stderr[:500]}")
    
    # Find the output JSON file
    json_files = list(output_dir.glob("*.json"))
    if not json_files:
        error_msg = f"No JSON output found in {output_dir}\n"
        error_msg += f"Directory contents: {list(output_dir.iterdir())}\n"
        error_msg += f"STDOUT:\n{result.stdout}\n"
        error_msg += f"STDERR:\n{result.stderr}"
        raise RuntimeError(error_msg)
    
    # Read the full result
    with open(json_files[0], 'r', encoding='utf-8') as f:
        full_result = json.load(f)

    if full_result.get("status") != "completed":
        failure = full_result.get("failure") or {}
        raise RuntimeError(
            "Search agent did not produce a terminal answer: "
            f"{failure.get('message', full_result.get('status', 'unknown status'))}"
        )
    
    # Prefer explicit final-answer fields, then support the legacy output trace.
    # Never send the serialized execution artifact to the judge as an answer.
    answer = None
    for key in ("predicted_answer", "final_answer", "answer"):
        value = full_result.get(key)
        if isinstance(value, str) and value.strip():
            answer = value
            break

    result_list = full_result.get("result", [])
    if answer is None and result_list:
        block = result_list[-1]
        if isinstance(block, dict) and block.get("type") == "output_text":
            for key in ("output", "text", "content"):
                value = block.get(key)
                if isinstance(value, str) and value.strip():
                    answer = value
                    break

    if answer is None:
        raise RuntimeError("Completed Search artifact has no non-empty final answer")
    
    # Extract tool call counts and retrieved docids
    tool_call_counts = full_result.get("tool_call_counts", {})
    retrieved_docids = full_result.get("retrieved_docids", [])
    
    return {
        "answer": answer,
        "trace": result_list,  # Full trace including all output blocks
        "tool_call_counts": tool_call_counts,
        "retrieved_docids": retrieved_docids,
        "full_result": full_result,
        "memory_context": memory_context,  # Retrieved memory context
        "query_with_memory": query_with_memory,  # Full query including memory context
    }

# Standalone function for loading correct answers
def load_correct_answers(query_id: str, data_dir: Path) -> Dict:
    """
    Load correct answers from data_dir/browsecomp_all_jsons.jsonl.

    The JSONL has one object per line: { "id", "question": [...], "answer": [...] }.
    question[i] is subquery i+1 question, answer[i] is subquery i+1 answer.
    The last index is the final query and final answer.
    """
    jsonl_path = data_dir / "browsecomp_all_jsons.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Cannot find browsecomp_all_jsons.jsonl at {jsonl_path}"
        )
    query_id_str = str(query_id)
    data = None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if str(obj.get("id", "")) == query_id_str:
                data = obj
                break
    if data is None:
        raise FileNotFoundError(
            f"Query id {query_id} not found in {jsonl_path}"
        )
    questions = data.get("question", data.get("questions", []))
    answers = data.get("answer", data.get("answers", []))
    if not questions or not answers:
        return {
            "query_id": query_id_str,
            "original_query": questions[-1] if questions else "",
            "subqueries": [],
            "correct_answers": [],
        }
    # Align lengths: use min length so we have a pair for each
    n = min(len(questions), len(answers))
    questions = questions[:n]
    answers = answers[:n]
    original_query = questions[-1] if questions else ""
    subquery_questions = questions[:-1] if len(questions) > 1 else []
    subquery_answers = answers[:-1] if len(answers) > 1 else []
    correct_answers = {
        "query_id": query_id_str,
        "original_query": original_query,
        "subqueries": subquery_questions,
        "correct_answers": [
            {"subquery": q, "correct_answer": a}
            for q, a in zip(subquery_questions, subquery_answers)
        ],
    }
    for idx, (q, a) in enumerate(zip(subquery_questions, subquery_answers), start=1):
        print(
            f"[DEBUG] Loaded correct answer | query {query_id} | index {idx}/{len(subquery_questions)}\n"
            f"  Subquery: {q}...\n"
            f"  Correct Answer: {a}...\n"
        )
    return correct_answers
