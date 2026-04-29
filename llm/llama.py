"""
SupplyAI LLM Module
Connects to local Llama3 via Ollama for inference.
"""

from langchain_community.llms import Ollama
from typing import Optional


class SupplyChainLLM:
    """
    Wrapper for Ollama-based Llama3 local LLM.
    Configured for supply chain reasoning and recommendations.
    """
    
    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        num_ctx: int = 4096
    ):
        """
        Initialize the LLM connection.
        
        Args:
            model_name: Ollama model name (e.g., 'llama3', 'llama3.1', 'llama3:8b')
            base_url: Ollama server URL
            temperature: Sampling temperature (lower = more deterministic)
            num_ctx: Context window size
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._llm: Optional[Ollama] = None
        
    @property
    def llm(self) -> Ollama:
        """
        Lazy-load the Ollama LLM instance.
        """
        if self._llm is None:
            print(f"[INFO] Connecting to Ollama model: {self.model_name}")
            self._llm = Ollama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                num_ctx=self.num_ctx
            )
        return self._llm
    
    def generate(self, prompt: str) -> str:
        """
        Generate text from a raw prompt string.
        
        Args:
            prompt: Full prompt text
            
        Returns:
            Generated response string
        """
        return self.llm.invoke(prompt)
    
    def is_available(self) -> bool:
        """
        Check if Ollama server is reachable and model is available.
        
        Returns:
            True if ready for inference
        """
        try:
            # Quick test invocation
            self.llm.invoke("Hi")
            return True
        except Exception as e:
            print(f"[ERROR] Ollama not available: {e}")
            return False