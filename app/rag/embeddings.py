"""
Embeddings Module

Handles the embedding model for converting text to vectors.
Uses sentence-transformers (local, free) by default.
"""
import os
from typing import List
import numpy as np


class EmbeddingModel:
    """Wrapper for embedding models."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Embedding model loaded successfully!")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = self.model.encode(texts, show_progress_bar=len(texts) > 10)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query string to embed
        
        Returns:
            Embedding vector
        """
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


# Singleton instance for efficiency
_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create the singleton embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


if __name__ == "__main__":
    # Test embeddings
    model = get_embedding_model()
    
    test_texts = [
        "What is the PTO policy?",
        "How many vacation days do I get?",
        "Password requirements for security"
    ]
    
    embeddings = model.embed_documents(test_texts)
    print(f"Embedding dimension: {model.dimension}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Sample embedding shape: {len(embeddings[0])}")
