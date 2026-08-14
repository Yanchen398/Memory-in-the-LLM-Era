# Method_memoryarena

Unified MemoryArena integrations for 12 memory methods:

`amem`, `memorybank`, `memgpt`, `mem0`, `mem0g`, `memochat`, `zep`,
`memtree`, `memoryos`, `memos`, `memgas`, and `lightmem`.

The common CLI is `run.py`. It can start the memory server or dispatch an
evaluation to the existing task runners with the requested memory method.

## Setup

Run all commands from the repository root.

```bash
pip install -r requirements.txt
```

Configure the model and embedding services required by the selected method,
for example:

```bash
export MEMORYARENA_LLM_BASE_URL=http://127.0.0.1:8002/v1
export MEMORYARENA_LLM_API_KEY=EMPTY
export MEMORYARENA_LLM_MODEL=Qwen3.5-9B
export MEMORYARENA_EMBEDDING_BASE_URL=http://127.0.0.1:8003/v1
export MEMORYARENA_EMBEDDING_MODEL=/path/to/embedding-model
export MEMORYARENA_STATE_ROOT=/path/to/memoryarena-state
```

Method-specific dependencies and environment variables are defined in each
`<method>/backend.py`.

## Start the Services

Start the task environment server:

```bash
python code/Method_memoryarena/runtime/env/env_server.py
```

Start the unified memory server in another terminal:

```bash
python code/Method_memoryarena/run.py serve \
  --host 127.0.0.1 \
  --port 8000 \
  --top-k 10 \
  --context-char-budget 8000
```

List the registered methods with:

```bash
python code/Method_memoryarena/run.py methods
```

## Run an Evaluation

Supported tasks are `math`, `physics`, `search`, `travel`, and `shopping`.
Run one method with:

```bash
python code/Method_memoryarena/run.py run \
  --task travel \
  --method mem0 \
  --config /path/to/travel_config.json \
  --memory-url http://127.0.0.1:8000 \
  --output-root results
```

The result is written below `results/travel/mem0/`. The `--task` option may be
omitted when the task can be inferred from `task_name` or `env.env_name` in the
configuration file.

Run all 12 methods sequentially with:

```bash
python code/Method_memoryarena/run.py run \
  --task search \
  --method all \
  --config /path/to/search_config.json \
  --memory-url http://127.0.0.1:8000 \
  --output-root results
```

`--output-root` is required for `--method all` so that every method receives a
separate output directory. Add `--fail-fast` to stop on the first failed method,
or `--dry-run` to inspect the dispatch without starting an evaluation.

To generate the complete 12-method, 5-task configuration matrix, see:

```bash
python code/Method_memoryarena/runtime/build_configs.py --help
```

The unified CLI overrides `memory.memory_system_name` and the memory server URL
in the supplied configuration. All other task, model, dataset, and evaluation
settings continue to come from that configuration file.
