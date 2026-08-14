"""
OpenAI embedding searcher implementation for dense retrieval.

This is the searcher used by the search tool when run_search / openai_client.py
is run with --searcher-type openai (default). It uses FAISS indexes + OpenAI-
compatible embeddings (OPENAI_API_KEY, OPENAI_BASE_URL) for query encoding.
"""

import glob
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import openai
from tqdm import tqdm

from .base import BaseSearcher

logger = logging.getLogger(__name__)


class OpenAISearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument(
            "--index-path",
            default="embeddings/shard*.index",
            help="Path to the FAISS index file (e.g., embeddings/shard*.index).",
        )
        parser.add_argument(
            "--id-map-path",
            default="embeddings/shard*_id_map.json",
            help="Path to the JSON file mapping index positions to document IDs (e.g., embeddings/shard*_id_map.json).",
        )
        parser.add_argument(
            "--corpus-path",
            default="corpus.jsonl",
            help="Path to the corpus JSONL file for retrieving document texts (e.g., corpus.jsonl).",
        )
        parser.add_argument(
            "--openai-model",
            default="text-embedding-ada-002",
            help="OpenAI embedding model to use (default: text-embedding-ada-002).",
        )

    # Chat/completion models that do NOT support embeddings (API returns 400 OperationNotSupported)
    _CHAT_MODEL_PATTERNS = ("gpt-4", "gpt-5", "o1-", "o3-", "claude-", "chatgpt")

    def __init__(self, args):
        self.args = args
        self.indexes: List[faiss.Index] = []
        self.id_maps: List[List[str]] = []
        self.docid_to_text = None
        # Use --openai-model for embeddings; if it looks like a chat model, use a real embedding model
        embedding_model = args.openai_model or ""
        model_lower = embedding_model.lower() if isinstance(embedding_model, str) else ""
        if any(p in model_lower for p in OpenAISearcher._CHAT_MODEL_PATTERNS):
            fallback = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
            fallback_lower = fallback.lower() if isinstance(fallback, str) else ""
            if any(p in fallback_lower for p in OpenAISearcher._CHAT_MODEL_PATTERNS):
                fallback = "text-embedding-ada-002"
                logger.warning(
                    "openai_model=%s and OPENAI_EMBEDDING_MODEL are chat models (not supported for embeddings). Using %s for embeddings.",
                    embedding_model,
                    fallback,
                )
            else:
                logger.warning(
                    "openai_model=%s looks like a chat model (not supported for embeddings). Using %s for embeddings.",
                    embedding_model,
                    fallback,
                )
            embedding_model = fallback
        self.openai_model = embedding_model
        self.provider = args.provider

        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            if not api_key:
                raise ValueError("OPENAI_API_KEY must be set for OpenAI provider")
        elif self.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            base_url = "https://openrouter.ai/api/v1"
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY must be set for OpenRouter provider")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Initialize OpenAI client (for embeddings in searcher)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url if base_url else None)
        self._embedding_base_url = base_url or "https://api.openai.com (default)"

        logger.info(
            "Initializing OpenAI searcher with provider=%s, embedding_base=%s, model=%s",
            self.provider,
            self._embedding_base_url,
            getattr(self.args, "openai_model", "?"),
        )

        self._load_faiss_indexes()
        self._load_id_maps()
        self._load_corpus()

        logger.info("OpenAI searcher initialized successfully")

    def _resolve_paths(self, pattern: str) -> List[str]:
        if any(ch in pattern for ch in ["*", "?", "["]):
            return sorted(glob.glob(pattern))
        return [pattern]

    def _load_faiss_indexes(self) -> None:
        index_paths = self._resolve_paths(self.args.index_path)
        if not index_paths:
            raise FileNotFoundError(f"Index file not found: {self.args.index_path}")
        for path in index_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Index file not found: {path}")
            index = faiss.read_index(path)
            self.indexes.append(index)
        total_vectors = sum(index.ntotal for index in self.indexes)
        logger.info(
            f"Loaded {len(self.indexes)} FAISS index shard(s) with {total_vectors} vectors"
        )

    def _load_id_maps(self) -> None:
        id_map_paths = self._resolve_paths(self.args.id_map_path)
        if not id_map_paths:
            raise FileNotFoundError(f"ID map file not found: {self.args.id_map_path}")
        if len(id_map_paths) != len(self.indexes):
            raise ValueError(
                "Number of id_map files does not match index shards: "
                f"{len(id_map_paths)} id maps vs {len(self.indexes)} indexes"
            )
        for path in id_map_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"ID map file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                ids = data.get("ids")
                if ids is None:
                    raise ValueError(f"ID map missing 'ids' key: {path}")
            elif isinstance(data, list):
                ids = data
            else:
                raise ValueError(f"Unexpected ID map format: {path}")
            self.id_maps.append(ids)
        total_ids = sum(len(ids) for ids in self.id_maps)
        logger.info(f"Loaded {len(self.id_maps)} ID map shard(s) with {total_ids} entries")

    def _load_corpus(self) -> None:
        if not os.path.exists(self.args.corpus_path):
            raise FileNotFoundError(f"Corpus file not found: {self.args.corpus_path}")
        self.docid_to_text = {}
        with open(self.args.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.docid_to_text[obj["docid"]] = obj["text"]
        logger.info(f"Loaded corpus with {len(self.docid_to_text)} documents")

    def _get_embedding(
        self,
        text: str,
        max_retries: int = 4,
        base_delay: float = 2.0,
        timeout: Optional[float] = 60.0,
    ) -> np.ndarray:
        """Call embedding API with retries for 503/429/connection errors."""
        last_error = None
        for attempt in range(max_retries):
            try:
                kwargs = {"input": [text], "model": self.openai_model}
                if timeout is not None:
                    kwargs["timeout"] = timeout
                response = self.client.embeddings.create(**kwargs)
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                last_error = e
                status = getattr(e, "status_code", None) or getattr(
                    getattr(e, "response", None), "status_code", None
                )
                is_retryable = (
                    status in (503, 429, 500)
                    or "503" in str(e)
                    or "429" in str(e)
                    or "connection" in str(e).lower()
                    or "timeout" in str(e).lower()
                )
                if is_retryable and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Embedding API error (attempt %d/%d): %s. Retrying in %.1fs ...",
                        attempt + 1,
                        max_retries,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Embedding API failed (base_url=%s, model=%s): %s",
                        getattr(self, "_embedding_base_url", "?"),
                        self.openai_model,
                        e,
                        exc_info=True,
                    )
                    raise
        raise last_error

    def search(self, query: str, k: int = 10, allowed_docids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if not self.indexes or not self.id_maps or not self.docid_to_text:
            raise RuntimeError("Searcher not properly initialized")

        query_emb = self._get_embedding(query)
        shard_results: List[Tuple[float, str]] = []

        for index, id_map in zip(self.indexes, self.id_maps):
            if allowed_docids is not None:
                allowed_set = set(allowed_docids)
                allowed_indices = [
                    i for i, docid in enumerate(id_map) if docid in allowed_set
                ]
                if not allowed_indices:
                    continue
                subset_vectors = np.zeros((len(allowed_indices), index.d), dtype=np.float32)
                for i, idx in enumerate(allowed_indices):
                    subset_vectors[i] = index.reconstruct(int(idx))
                temp_index = faiss.IndexFlatL2(index.d)
                temp_index.add(subset_vectors)
                scores, indices = temp_index.search(
                    query_emb.reshape(1, -1), min(k, len(allowed_indices))
                )
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:
                        continue
                    original_idx = allowed_indices[idx]
                    docid = id_map[original_idx]
                    shard_results.append((float(-score), docid))
            else:
                scores, indices = index.search(query_emb.reshape(1, -1), k)
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:
                        continue
                    docid = id_map[idx]
                    shard_results.append((float(-score), docid))

        shard_results.sort(key=lambda item: item[0], reverse=True)
        top_results = shard_results[:k]

        results: List[Dict[str, Any]] = []
        for score, docid in top_results:
            text = self.docid_to_text.get(docid, "Text not found")
            results.append({"docid": docid, "score": score, "text": text})

        return results

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        if not self.docid_to_text:
            raise RuntimeError("Corpus not loaded")

        text = self.docid_to_text.get(docid)
        if text is None:
            return None

        return {
            "docid": docid,
            "text": text,
        }

    @property
    def search_type(self) -> str:
        return "OpenAI_FAISS"
