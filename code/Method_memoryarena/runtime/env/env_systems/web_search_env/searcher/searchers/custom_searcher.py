import os
import json
import logging
from typing import Any, Dict, Optional, List
import numpy as np
import openai
import faiss

from .base import BaseSearcher
print("ENV OPENAI_API_KEY:", repr(os.getenv("OPENAI_API_KEY")))
logger = logging.getLogger(__name__)
class CustomSearcher(BaseSearcher):
    """
    OpenAI embedding-based dense retriever using text-embedding-3-small
    """

    # -----------------------------
    # CLI ARGUMENTS
    # -----------------------------
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument(
            "--corpus-path",
            type=str,
            required=True,
            help="Path to corpus (jsonl or txt)",
        )
        parser.add_argument(
            "--index-path",
            type=str,
            required=True,
            help="Directory to store/load embeddings",
        )
        parser.add_argument(
            "--rebuild-index",
            action="store_true",
            help="Rebuild embedding index from corpus",
        )
        parser.add_argument(
            "--embedding-model",
            type=str,
            default="text-embedding-3-small",
            help="OpenAI embedding model",
        )

    # -----------------------------
    # INITIALIZATION
    # -----------------------------
    def __init__(self, args):
        self.args = args
        self.corpus_path = args.corpus_path
        self.index_path = args.index_path
        self.embedding_model = args.embedding_model

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment")
        self.client = openai.OpenAI(api_key=api_key,base_url="")

        os.makedirs(self.index_path, exist_ok=True)

        self.emb_path = os.path.join(self.index_path, "embeddings.npy")
        self.meta_path = os.path.join(self.index_path, "documents.json")
        self.faiss_path = os.path.join(self.index_path, "faiss.index")

        if (
            args.rebuild_index
            or not os.path.exists(self.emb_path)
            or not os.path.exists(self.faiss_path)
        ):
            logger.info("Building embedding index and FAISS index...")
            self._build_index()
        else:
            logger.info("Loading existing embedding and FAISS index...")
            self._load_index()

    # -----------------------------
    # INDEXING
    # -----------------------------
    def _load_corpus(self) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []

        if self.corpus_path.endswith(".jsonl"):
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    docs.append(json.loads(line))
        else:
            raise ValueError("Only jsonl corpus supported")

        return docs

    # ---- NEW: safe batched embedding for corpus OFFLINE ----
    def _embed_texts_batched(
        self,
        texts: List[str],
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Create embeddings for a list of texts (corpus building).
        Validates and batches input to avoid $.input errors.
        """
        if not texts:
            raise ValueError("Cannot create embeddings: input text list is empty.")

        clean: List[str] = []
        for t in texts:
            if not isinstance(t, str):
                raise TypeError(f"Embedding input must be str, got {type(t)}")
            s = t.strip()
            if not s:
                raise ValueError("Embedding input contains empty string.")
            clean.append(s)

        all_embs: List[np.ndarray] = []
        for i in range(0, len(clean), batch_size):
            batch = clean[i : i + batch_size]
            logger.info(
                f"Embedding corpus batch {i}–{i + len(batch) - 1} / {len(clean) - 1}"
            )
            resp = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )
            batch_emb = np.array(
                [item.embedding for item in resp.data], dtype=np.float32
            )
            all_embs.append(batch_emb)

        embeddings = np.concatenate(all_embs, axis=0)
        return embeddings

    def _build_index(self):
        docs = self._load_corpus()

        # ---- NEW: clean docs + texts before sending to OpenAI ----
        filtered_docs: List[Dict[str, Any]] = []
        filtered_texts: List[str] = []
        for d in docs:
            raw = d.get("text", d.get("content", ""))
            if not isinstance(raw, str):
                logger.warning(f"Skipping doc with non-string text: {d.get('docid')}")
                continue
            s = raw.strip()
            if not s:
                logger.warning(f"Skipping doc with empty text: {d.get('docid')}")
                continue
            filtered_docs.append(d)
            filtered_texts.append(s)

        if not filtered_texts:
            raise ValueError("No valid texts found in corpus for embeddings.")

        # Call OpenAI embeddings only once for corpus, with cleaned input
        embeddings = self._embed_texts_batched(filtered_texts)
        embeddings = self._normalize(embeddings)

        # Save only filtered docs and embeddings, so indices line up
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(filtered_docs, f)

        np.save(self.emb_path, embeddings)

        # Build and save FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, self.faiss_path)

        self.documents = filtered_docs
        self.embeddings = embeddings
        self.index = index

        logger.info(
            f"Indexed {len(filtered_docs)} documents and built FAISS index"
        )

    def _load_index(self):
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        self.embeddings = np.load(self.emb_path)
        self.index = faiss.read_index(self.faiss_path)

    # -----------------------------
    # SEARCH
    # -----------------------------
    def _embed_query(self, query: str) -> np.ndarray:
        """
        Lightweight embedding call for queries only.
        """
        q = query.strip()
        if not q:
            raise ValueError("Query text is empty.")
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=[q],
        )
        emb = np.array(resp.data[0].embedding, dtype=np.float32)
        return emb

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        # Only embed the query online
        query_emb = self._embed_query(query)
        query_emb = self._normalize(query_emb[np.newaxis, :])[0]

        # Use FAISS for search
        D, I = self.index.search(query_emb[np.newaxis, :], k)

        results: List[Dict[str, Any]] = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            results.append(
                {
                    "docid": doc["docid"],
                    "score": float(score),
                    "text": doc.get("text") or doc.get("content", ""),
                }
            )

        return results

    # -----------------------------
    # DOCUMENT FETCH
    # -----------------------------
    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        for doc in self.documents:
            if doc.get("docid") == docid:
                return doc
        return None

    # -----------------------------
    # UTILS
    # -----------------------------
    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    # -----------------------------
    # SEARCHER METADATA
    # -----------------------------
    @property
    def search_type(self) -> str:
        return "custom"

    def search_description(self, k: int = 10) -> str:
        return (
            f"Semantic search over a custom corpus using OpenAI embeddings. "
            f"Returns top-{k} documents with docid, relevance score, and text snippet."
        )

    def get_document_description(self) -> str:
        return "Retrieve the full document text by document ID."


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    CustomSearcher.parse_args(parser)
    args = parser.parse_args()

    CustomSearcher(args)

