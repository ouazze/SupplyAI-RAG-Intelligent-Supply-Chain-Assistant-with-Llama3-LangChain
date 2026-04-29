"""
SupplyAI RAG Chain Module
Orchestrates retrieval and generation for supply chain queries.
"""

from typing import Optional, Dict, Any
from langchain.chains import RetrievalQA
from langchain_core.callbacks import CallbackManagerForChainRun

from llm.llama import SupplyChainLLM
from vectorstore.faiss_store import SupplyChainVectorStore
from prompts.supply_prompt import SupplyChainPrompts


class SupplyChainRAGChain:
    """
    End-to-end RAG chain for supply chain intelligence.
    Combines FAISS retrieval with Llama3 generation using custom prompts.
    """
    
    def __init__(
        self,
        vector_store: SupplyChainVectorStore,
        llm: Optional[SupplyChainLLM] = None,
        prompt_type: str = "rag"
    ):
        """
        Initialize the RAG chain.
        
        Args:
            vector_store: Initialized FAISS vector store
            llm: LLM instance (creates default if None)
            prompt_type: Type of prompt to use ('rag', 'restock', 'risk')
        """
        self.vector_store = vector_store
        self.llm = llm or SupplyChainLLM()
        self.prompt_type = prompt_type
        self.qa_chain: Optional[RetrievalQA] = None
        self._setup_chain()
        
    def _get_prompt(self):
        """Select the appropriate prompt template."""
        prompts = SupplyChainPrompts()
        
        if self.prompt_type == "restock":
            return prompts.get_restock_recommendation_prompt()
        elif self.prompt_type == "risk":
            return prompts.get_risk_assessment_prompt()
        else:
            return prompts.get_rag_prompt()
    
    def _setup_chain(self) -> None:
        """
        Configure the RetrievalQA chain with custom prompt and retriever.
        """
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 most relevant records
        )
        
        prompt = self._get_prompt()
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm.llm,
            chain_type="stuff",  # Simple concatenation of retrieved docs
            retriever=retriever,
            return_source_documents=True,  # Include sources for transparency
            chain_type_kwargs={
                "prompt": prompt,
                "document_separator": "\n---\n"  # Clear separation between records
            },
            verbose=False
        )
        
        print(f"[INFO] RAG chain initialized with prompt type: {self.prompt_type}")
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute a query through the RAG pipeline.
        
        Args:
            question: Natural language question about supply chain
            
        Returns:
            Dictionary with 'answer', 'source_documents', and 'query'
        """
        if self.qa_chain is None:
            raise RuntimeError("RAG chain not initialized.")
            
        print(f"[INFO] Processing query: {question}")
        
        result = self.qa_chain.invoke({"query": question})
        
        # Format response for downstream use
        response = {
            "query": question,
            "answer": result.get("result", "No answer generated."),
            "source_documents": result.get("source_documents", []),
            "sources_count": len(result.get("source_documents", []))
        }
        
        return response
    
    def get_sources_metadata(self, response: Dict[str, Any]) -> list:
        """
        Extract clean metadata from source documents for UI display.
        
        Args:
            response: Response dict from query()
            
        Returns:
            List of source metadata dictionaries
        """
        sources = []
        for doc in response.get("source_documents", []):
            sources.append({
                "product": doc.metadata.get("product_name", "Unknown"),
                "product_id": doc.metadata.get("product_id", "N/A"),
                "category": doc.metadata.get("category", "N/A"),
                "stock": doc.metadata.get("current_stock", "N/A"),
                "supplier": doc.metadata.get("supplier", "N/A"),
                "warehouse": doc.metadata.get("warehouse", "N/A")
            })
        return sources