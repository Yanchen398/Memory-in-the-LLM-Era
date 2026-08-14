from __future__ import annotations

from typing import List, Optional

import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class TextEmbedder:
    _SBERT_MODEL_MAP = {
        "mpnet": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "minilm": "/path/to/local/all-MiniLM-L6-v2",
        "qaminilm": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    }

    def __init__(
        self,
        backend: str = "contriever",
        device: Optional[str] = None,
        batch_size: int = 64,
        api_key: str = "EMPTY",
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: int = 256,
    ) -> None:
        self.backend = backend
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim: Optional[int] = None
        self.api_model_name = model_name
        self.max_tokens = max_tokens
        self.client = None

        if backend == "contriever":
            self.model = AutoModel.from_pretrained("facebook/contriever").to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
            self.model.eval()
        elif backend in self._SBERT_MODEL_MAP:
            self.model = SentenceTransformer(self._SBERT_MODEL_MAP[backend], device=self.device)
            self.tokenizer = None
        elif backend == "openai":
            if not base_url or not model_name:
                raise ValueError("The openai embedder requires base_url and model_name.")
            self.client = OpenAI(
                api_key=api_key or "EMPTY",
                base_url=base_url,
                max_retries=5,
                timeout=120.0,
            )
            self.model = None
            self.tokenizer = None
        else:
            raise ValueError(
                f"Unsupported embedder '{backend}'. "
                f"Choose one of {['contriever', 'mpnet', 'minilm', 'qaminilm', 'openai']}."
            )

    def encode(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            if self.embedding_dim is None:
                return torch.empty((0, 0), dtype=torch.float32)
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)

        if self.backend == "contriever":
            vectors = self._encode_contriever(texts)
        elif self.backend == "openai":
            vectors = self._encode_openai(texts)
        else:
            vectors = self._encode_sbert(texts)
        self.embedding_dim = vectors.shape[1]
        return vectors

    def _encode_sbert(self, texts: List[str]) -> torch.Tensor:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        if not isinstance(vectors, torch.Tensor):
            vectors = torch.tensor(vectors)
        return vectors.detach().cpu().float()

    def _encode_openai(self, texts: List[str]) -> torch.Tensor:
        all_vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(
                model=self.api_model_name,
                input=batch,
                extra_body={"truncate_prompt_tokens": self.max_tokens},
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            if len(ordered) != len(batch):
                raise RuntimeError(
                    "Embedding API returned "
                    f"{len(ordered)} vectors for a batch of {len(batch)} texts."
                )
            all_vectors.extend(item.embedding for item in ordered)
        return torch.tensor(all_vectors, dtype=torch.float32)

    def _encode_contriever(self, texts: List[str]) -> torch.Tensor:
        all_vecs = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs).last_hidden_state
                pooled = self._mean_pooling(outputs, inputs["attention_mask"])
                all_vecs.append(pooled.cpu())
        return torch.cat(all_vecs, dim=0).float()

    @staticmethod
    def _mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = token_embeddings.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return token_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
