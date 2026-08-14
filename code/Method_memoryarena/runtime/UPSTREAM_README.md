# MemoryArena

Minimal, modular framework for connecting agents, environments, and memory.

Implementation for **MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks** (https://arxiv.org/abs/2602.16313). 

*This code is preview version. We are still actively maintaining and improving this codebase.*
## Quick Start 
**[Important]** Check `setup_web_shopping.md`, `setup_travel.md`, `setup_web_search_env.md`, and `setup_formal_reasoning.md`to follow step-by-step structions for each test environment.

## Repository Structure

- `agent/`: task agent implementations.
- `env/`: environment server, client, and environment systems.
- `memory/`: memory client and memory systems including long-context, letta, mirix, mem0, mem0-g, ReasoningBank, BM25, Text-embedding RAG,  GraphRAG,  and MemoRAG. 

## API Keys
- Make sure you have your `OPENAI_API_KEY`, `OPENAI_BASE_URL`,  `GOOGLE_API_KEY`,  `ANTHROPIC_API_KEY`,  `OPENROUTER_API_KEY`, `OPENROUTER_API_BASE_URL` set ready in either your bashrc file or in configs following each setup `md`.
- For Letta, Mirix, Mem0 (including Mem0-g), make sure you have their memory system api keys ready `LETTA_API_KEY`, `MIRIX_API_KEY`, `MEM0_API_KEY` in your bashrc file. 

## Example Flow

1. Task prompt → memory wraps prompt
2. Agent generates action
3. Env `step()` executes tool or accepts final
4. Observation + reward returned
5. Memory stores action/observation/reward(optional)

# Cite Our Paper:
If you are using this repo, please cite our paper at:
```
@article{he2026memoryarena,
  title={MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks},
  author={He, Zexue and Wang, Yu and Zhi, Churan and Hu, Yuanzhe and Chen, Tzu-Ping and Yin, Lang and Chen, Ze and Wu, Tong Arthur and Ouyang, Siru and Wang, Zihan and others},
  journal={arXiv preprint arXiv:2602.16313},
  year={2026}
}
```

