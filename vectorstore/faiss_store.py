"""
SupplyAI Vector Store Module
Manages FAISS index creation, persistence, and retrieval.
"""

import os
from pathlib import Path
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from embeddings.embedder import SupplyChainEmbedder


class SupplyChainVectorStore:
    """
    FAISS-based vector store for supply chain documents.
    Supports persistent storage to avoid re-embedding on every run.
    """
    
    def __init__(
        self, 
        index_path: str = "vectorstore/faiss_index",
        embedder: Optional[SupplyChainEmbedder] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            index_path: Directory path to save/load FAISS index
            embedder: Embedding model instance (creates default if None)
        """
        self.index_path = Path(index_path)
        self.embedder = embedder or SupplyChainEmbedder()
        self.vectorstore: Optional[FAISS] = None
        
    def build_index(self, documents: List[Document]) -> FAISS:
        """
        Build a new FAISS index from documents.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            FAISS vector store instance
        """
        print(f"[INFO] Building FAISS index with {len(documents)} documents...")
        
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedder.embeddings
        )
        
        # Persist to disk
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.index_path))
        print(f"[INFO] FAISS index saved to {self.index_path}")
        
        return self.vectorstore
    
    def load_index(self) -> Optional[FAISS]:
        """
        Load an existing FAISS index from disk.
        
        Returns:
            FAISS vector store or None if not found
        """
        if not self.index_path.exists():
            print(f"[INFO] No existing index found at {self.index_path}")
            return None
            
        print(f"[INFO] Loading existing FAISS index from {self.index_path}")
        self.vectorstore = FAISS.load_local(
            str(self.index_path),
            self.embedder.embeddings,
            allow_dangerous_deserialization=True  # Safe for local files only
        )
        return self.vectorstore
    
    def get_or_create(self, documents: List[Document]) -> FAISS:
        """
        Load existing index if available, otherwise build from documents.
        
        Args:
            documents: Documents to index if no existing index
            
        Returns:
            FAISS vector store instance
        """
        existing = self.load_index()
        if existing is not None:
            return existing
            
        return self.build_index(documents)
    
    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a retriever interface for the vector store.
        
        Args:
            search_kwargs: Retrieval parameters (e.g., {"k": 5})
            
        Returns:
            BaseRetriever instance
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call build_index or load_index first.")
            
        kwargs = search_kwargs or {"k": 5}
        return self.vectorstore.as_retriever(search_kwargs=kwargs)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Direct similarity search against the index.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant Document objects
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized.")
            
        return self.vectorstore.similarity_search(query, k=k)