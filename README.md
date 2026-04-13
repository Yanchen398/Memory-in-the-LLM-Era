# Memory in the LLM Era: Modular Architectures and Strategies within a Unified Framework

This repository contains the supplementary materials for our VLDB submission, including the appendix, source code, and plotting notebooks for the figures reported in the paper.

In this paper, we propose a unified modular framework that decomposes existing agent memory systems into four components: *Information Extraction*, *Memory Management*, *Memory Storage*, and *Information Retrieval*. We systematically benchmark representative memory methods together with our newly designed state-of-the-art method on long-term conversational datasets.

## Repository Overview

- `appendix.pdf`: supplementary appendix of the paper.
- `code/`: source code and detailed instructions for running experiments and reproducing the main results.
- `figure_plot.ipynb`: Jupyter notebooks for plotting the figures appearing in the paper. For convenience, the final results used for plotting are directly included in these notebooks. The corresponding raw results can be obtained by running the code in `code/`.

## How to Navigate This Repository

If you would like to read the supplementary technical details, please start with `appendix.pdf`.

If you would like to reproduce the experimental results, please continue to the code and reproduction guide below.

If you would like to inspect or regenerate the figures in the paper, please open the provided Jupyter notebook in the repository root.

## Code and Reproduction Guide

The `code/` directory contains the official implementation for the paper.

### Supported Memory Methods

We implement and evaluate the following memory architectures under our unified framework:

| Method | Paper | Code |
|--------|-------|------|
| **A-MEM** | [arXiv 2502.12110](https://arxiv.org/abs/2502.12110) | [GitHub](https://github.com/WujiangXu/AgenticMemory) |
| **MemoryBank** | [arXiv 2305.10250](https://arxiv.org/abs/2305.10250) | [GitHub](https://github.com/zhongwanjun/MemoryBank-SiliconFriend) |
| **MemGPT (Letta)** | [arXiv 2310.08560](https://arxiv.org/abs/2310.08560) | [GitHub](https://github.com/letta-ai/letta) |
| **Mem0 / Mem0$^g$** | [arXiv 2504.19413](https://arxiv.org/abs/2504.19413) | [GitHub](https://github.com/mem0ai/mem0) |
| **MemoChat** | [arXiv 2308.08239](https://arxiv.org/abs/2308.08239) | [GitHub](https://github.com/LuJunru/MemoChat) |
| **Zep (Graphiti)** | [arXiv 2501.13956](https://arxiv.org/abs/2501.13956) | [GitHub](https://github.com/getzep/graphiti) |
| **MemTree** | [arXiv 2410.14052](https://arxiv.org/abs/2410.14052) | *(no official implementation)* |
| **MemoryOS** | [arXiv 2506.06326](https://arxiv.org/abs/2506.06326) | [GitHub](https://github.com/BAI-LAB/MemoryOS) |
| **MemOS** | [arXiv 2507.03724](https://arxiv.org/abs/2507.03724) | [GitHub](https://github.com/MemTensor/MemOS) |

### Installation

#### Requirements

- Python 3.10+
- An OpenAI-compatible API endpoint
- Neo4j graph database (for Mem0$^g$, Zep and MemOS)
- PostgreSQL 14 with pgvector enabled (for MemGPT)

#### Get the Code

```bash
git clone https://github.com/Yanchen398/Memory-in-the-LLM-Era.git
cd Memory-in-the-LLM-Era/code
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

### Project Structure

```text
code/
├── run.py                  # Unified entry point for all methods
├── eval.py                 # Unified evaluation script
├── simplify.py             # LLM-assisted answer simplification
├── utils.py                # ConfigManager and shared utilities
├── Config/                 # YAML config files for each method
│   ├── amem.yaml
│   ├── mem0.yaml
│   ├── memgpt.yaml
│   ├── memochat.yaml
│   ├── memorybank.yaml
│   ├── memoryos.yaml
│   ├── memos.yaml
│   ├── memtree.yaml
│   ├── sota.yaml
│   └── zep.yaml
├── Method/                 # Method implementations
│   ├── amem/
│   ├── mem0/
│   ├── memgpt/
│   ├── memochat/
│   ├── memorybank/
│   ├── memoryos/
│   ├── memos/
│   ├── memtree/
│   ├── sota/
│   └── zep/
├── Dataset/                # Dataset files (not included, see below)
│   ├── LOCOMO/
│   └── LONGMEMEVAL/
└── Result/                 # Output results (auto-created)
    ├── LOCOMO/
    └── LONGMEMEVAL/
```

### Datasets

Place dataset files under `code/Dataset/`:

```text
code/Dataset/
├── LOCOMO/
│   └── locomo10.json
└── LONGMEMEVAL/
    ├── longmemeval_s.json
    └── 5_filler_sess/
        └── data_5_filler_sess.json   # used to generate the "Context scalability analysis" variant dataset
```

#### Dataset Preparation

- **LOCOMO**: download the dataset from <https://github.com/snap-research/locomo/tree/main/data> and use it directly.
- **LongMemEval (LME)**: the dataset download link can be found at <https://github.com/xiaowu0162/LongMemEval/tree/main>. After downloading the dataset, run `code/Dataset/LONGMEMEVAL/transform.py` to convert it into the LOCOMO-aligned format used by this repository.
- **LongMemEval variants for analysis**: for the position sensitivity analysis and context scalability analysis variants, first run `code/Dataset/LONGMEMEVAL/gen_variants.py` to generate the variant files, and then run `code/Dataset/LONGMEMEVAL/transform.py` on those generated variants as well.

#### Dataset JSON Format

All datasets used by this repository are expected to follow a unified schema that is aligned with the LOCOMO format. LongMemEval and all LongMemEval variants should be converted to this format before use:

```text
[
  {
    "sample_id": "sample_001",
    "qa": [
      {
        "question": "When did Caroline go to the LGBTQ support group?",
        "answer": "7 May 2023",
        "evidence": [
          "D1:3"
        ],
        "category": 2
      },
      {
        ...(more questions)
      }
    ],
    "conversation": {
      "speaker_a": "Caroline",
      "speaker_b": "Melanie",
      "session_1_date_time": "1:56 pm on 8 May, 2023",
      "session_1": [
        {
          "speaker": "Caroline",
          "dia_id": "D1:1",
          "text": "Hey Mel! Good to see you! How have you been?"
        },
        {
          ...(more dialogues)
        }
      ],
      ...(more sessions)
    }
  },
]
```

### Configuration

All method parameters are configured in YAML files under `code/Config/`. Before running a method, edit its corresponding config file and set dataset paths, output paths, model parameters, and method-specific options there. For other methods, you can follow the existing config examples provided in the `code/Config/` folder.

#### Config Example

**`code/Config/memoryos.yaml`**

```yaml
dataset_path: ./Dataset/LOCOMO/locomo10.json
memory_path: ./Result/LOCOMO/memoryos/locomo10test/mem_data
output_path: ./Result/LOCOMO/memoryos/locomo10test/result.json
llm_model: your_chat_model_name
llm_api_key: your_llm_api_key
llm_base_url: your_llm_base_url
embedding_model_name: your_embedding_model_name
```

### Running Methods

All methods are launched through the unified entry point. In normal usage, `run.py` only needs the method name and the path to its YAML config file:

```bash
cd code
python run.py <method> --config_file <config.yaml>
```

#### Running Example

```bash
cd code
python run.py memtree --config_file Config/memtree.yaml
```

### Result Simplification

As discussed in the paper, F1 and BLEU-1 metrics are highly sensitive to answer verbosity. Before evaluation, run the answer simplification script, which uses an LLM to extract the core information from verbose responses.

```bash
cd code
python simplify.py --dataset <dataset> --method <method> --version <version> \
  --model <chat-model> \
  --base_url <api-endpoint> \
  --api_key <api-key>
```

- `--dataset`: benchmark name. Use `loco` for LOCOMO and `lme` for LONGMEMEVAL.
- `--method`: method name, such as `memtree`, `memoryos`.
- `--version`: result version identifier, for example `testv1`.
- `--model`: chat model used for simplification.
- `--base_url`: API endpoint for the model service.
- `--api_key`: API key for the model service.

Given `--dataset <dataset>`, `--method <method>`, and `--version <version>`, the script reads the raw result file from:

```text
Result/{BENCHMARK}/{method}/{version}/result.json
```

and writes the simplified output to:

```text
Result/{BENCHMARK}/{method}/{version}/result_simplified.json
```

For example:

```bash
python simplify.py --dataset loco --method memtree --version new \
  --model your_llm_model_name \
  --base_url your_llm_base_url \
  --api_key your_llm_api_key
```

Input: `Result/LOCOMO/memtree/new/result.json`

Output: `Result/LOCOMO/memtree/new/result_simplified.json`

### Evaluation

Use `eval.py` with the same parameters `--dataset`, `--method`, `--version`, and `embedding_model`:

```bash
python eval.py --dataset <dataset> --method <method> --version <version> \
  --embedding_model <embedding-model>
```

Input: `Result/{BENCHMARK}/{method}/{version}/result_simplified.json`

Outputs:

- `Result/{BENCHMARK}/{method}/{version}/{method}_{benchmark}_judged.json`
- `Result/{BENCHMARK}/{method}/{version}/{method}_{benchmark}_statistics.txt`
- `Result/{BENCHMARK}/{method}/{version}/{method}_{benchmark}_statistics.json`

### Full Pipeline Example

End-to-end example using MemTree on LOCOMO:

```bash
cd code

# 1. Run the memory method (set output path to ./Result/LOCOMO/memtree/new/result.json in config)
python run.py memtree --config_file Config/memtree.yaml

# 2. Simplify the raw responses
python simplify.py --dataset loco --method memtree --version new \
  --model your_llm_model_name \
  --base_url your_llm_base_url \
  --api_key your_llm_api_key

# 3. Evaluate
python eval.py --dataset loco --method memtree --version new \
  --embedding_model /path/to/embedding-model
```

### Token Usage Tracking

Each method includes a built-in `TokenTracker` that records LLM token consumption broken down by sample and stage. Output is written to a JSON file configured per method.

This enables cost estimation and token-efficiency comparison across methods.

## License

This project is released under the [MIT License](code/LICENSE).
