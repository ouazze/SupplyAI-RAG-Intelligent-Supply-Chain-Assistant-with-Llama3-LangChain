"""
SupplyAI Embedding Module
Handles text vectorization using HuggingFace sentence-transformers.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional


class SupplyChainEmbedder:
    """
    Wrapper for HuggingFace embeddings optimized for supply chain semantic search.
    Uses all-MiniLM-L6-v2 for fast, high-quality local embeddings.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model identifier for embeddings
        """
        self.model_name = model_name
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """
        Lazy-load and cache the embedding model.
        """
        if self._embeddings is None:
            print(f"[INFO] Loading embedding model: {self.model_name}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU available
                encode_kwargs={'normalize_embeddings': True}
            )
        return self._embeddings
    
    def embed_query(self, text: str) -> list:
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: list) -> list:
        return self.embeddings.embed_documents(texts)