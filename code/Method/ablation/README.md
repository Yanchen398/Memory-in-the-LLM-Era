# Ablation Experiments

This package provides one CLI for the core 2x2 ablation matrix, graph/tree
re-retrieval, and the MemoryOS segment-granularity ablation.

Run commands from the repository root with `code` on `PYTHONPATH`:

```bash
export PYTHONPATH=code
export ABLATION_DATASET_PATH=/path/to/locomo.json
export ABLATION_OUTPUT_ROOT=/path/to/ablation-results
export ABLATION_LLM_MODEL=your-model
export ABLATION_LLM_API_KEY=EMPTY
export ABLATION_LLM_BASE_URL=https://your-llm-endpoint/v1
export ABLATION_LLM_BASE_URLS=https://endpoint-a/v1,https://endpoint-b/v1
export ABLATION_EMBEDDING_MODEL=/path/to/embedding-model
export ABLATION_EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1
```

Run one core combination:

```bash
python -m Method.ablation run \
  --config-path code/Method/ablation/configs/locomo_qwen35_9b.yaml \
  --memory-granularity segment \
  --mid-term-structure tree
```

Run all four `segment/message x tree/graph` combinations sequentially:

```bash
python -m Method.ablation matrix \
  --config-path code/Method/ablation/configs/locomo_qwen35_9b.yaml
```

Re-query persisted graph or tree indexes:

```bash
python -m Method.ablation graph-reretrieve \
  --config-path code/Method/ablation/configs/locomo_qwen35_9b.yaml \
  --source-mode segment_graph \
  --output-dir /path/to/graph-reretrieve-results

python -m Method.ablation tree-reretrieve \
  --config-path code/Method/ablation/configs/locomo_qwen35_9b.yaml \
  --source-mode segment_tree \
  --output-dir /path/to/tree-reretrieve-results \
  --dialogue-top-k 10 \
  --other-memory-top-k 10
```

Run the MemoryOS segment ablation:

```bash
python -m Method.ablation memoryos-segment \
  --config-path code/Method/ablation/configs/memoryos_segment_locomo_qwen35_9b.yaml
```

Use `python -m Method.ablation --help` for all commands and options. The sample
configs contain environment-variable placeholders only; they do not include
machine-specific paths, credentials, or service addresses.
