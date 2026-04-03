from sentence_transformers import SentenceTransformer

from memos.configs.embedder import SenTranEmbedderConfig
from memos.embedders.base import BaseEmbedder
from memos.log import get_logger


logger = get_logger(__name__)


class SenTranEmbedder(BaseEmbedder):
    """Sentence Transformer Embedder class."""

    def __init__(self, config: SenTranEmbedderConfig):
        self.config = config
        self.model = SentenceTransformer(
            self.config.model_name_or_path, trust_remote_code=self.config.trust_remote_code
        )

        if self.config.embedding_dims is not None:
            logger.warning(
                "SentenceTransformer does not support specifying embedding dimensions directly. "
                "The embedding dimension is determined by the model."
                "`embedding_dims` will be ignored."
            )
            # Get embedding dimensions from the model
            self.config.embedding_dims = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for the given texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings, each represented as a list of floats.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
