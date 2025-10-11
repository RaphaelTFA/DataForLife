from sentence_transformers import SentenceTransformer
import numpy as np
from math_rag_pipeline.src.rag_toan.config import DEFAULT_EMBEDDER

class Embedder:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or DEFAULT_EMBEDDER
        self.model = SentenceTransformer(self.model_name)

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        """
        Returns numpy array (n, dim)
        """
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        return embeddings
