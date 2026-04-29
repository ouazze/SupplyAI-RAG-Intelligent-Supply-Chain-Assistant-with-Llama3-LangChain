"""
SupplyAI Main Entry Point
Backend runner for command-line interaction and system testing.
"""

import os
import sys
from pathlib import Path

from data_loader.loader import SupplyChainDataLoader
from embeddings.embedder import SupplyChainEmbedder
from vectorstore.faiss_store import SupplyChainVectorStore
from llm.llama import SupplyChainLLM
from chains.rag_chain import SupplyChainRAGChain


def check_ollama():
    """Verify Ollama is running and Llama3 is available."""
    print("🔍 Checking Ollama availability...")
    llm = SupplyChainLLM()
    if llm.is_available():
        print("✅ Ollama is running and model is accessible.\n")
        return True
    else:
        print("❌ Ollama not available. Please ensure:")
        print("   1. Ollama is installed: https://ollama.com")
        print("   2. Ollama service is running")
        print("   3. Model is pulled: ollama pull llama3")
        return False


def build_system(csv_path: str):
    """
    Build the complete SupplyAI pipeline.
    
    Args:
        csv_path: Path to supply chain CSV file
    """
    print("🏭 SupplyAI - Supply Chain Intelligence System")
    print("=" * 50)
    
    # Step 1: Load Data
    print("\n📊 Step 1: Loading supply chain data...")
    loader = SupplyChainDataLoader(csv_path)
    documents = loader.to_documents()
    summary = loader.get_inventory_summary()
    
    print(f"   • Total products: {summary['total_products']}")
    print(f"   • Low stock items: {summary['low_stock_items']}")
    print(f"   • At-risk items: {summary['at_risk_items']}")
    
    # Step 2: Build Vector Store
    print("\n🔎 Step 2: Initializing embeddings and vector store...")
    embedder = SupplyChainEmbedder()
    vector_store = SupplyChainVectorStore(embedder=embedder)
    vector_store.get_or_create(documents)
    
    # Step 3: Setup LLM
    print("\n🧠 Step 3: Connecting to Llama3 via Ollama...")
    llm = SupplyChainLLM()
    
    # Step 4: Build RAG Chain
    print("\n🔗 Step 4: Building RAG chain...")
    rag_chain = SupplyChainRAGChain(
        vector_store=vector_store,
        llm=llm,
        prompt_type="rag"
    )
    
    print("\n" + "=" * 50)
    print("✅ SupplyAI system initialized successfully!")
    print("Type 'quit' to exit, 'risk' for risk assessment mode.\n")
    
    return rag_chain


def interactive_mode(rag_chain: SupplyChainRAGChain):
    """
    Run interactive CLI session.
    """
    while True:
        try:
            question = input("\n📝 Ask SupplyAI: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not question:
                continue
            
            # Execute query
            response = rag_chain.query(question)
            
            # Display results
            print("\n" + "=" * 50)
            print("🤖 SUPPLYAI RESPONSE:")
            print("=" * 50)
            print(response["answer"])
            print("=" * 50)
            
            # Show sources
            sources = rag_chain.get_sources_metadata(response)
            if sources:
                print(f"\n📎 Based on {len(sources)} supply records:")
                for src in sources:
                    print(f"   • {src['product']} (Stock: {src['stock']}, Warehouse: {src['warehouse']})")
                    
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    # Default CSV path (matches user requirement)
    default_path = r"C:\Users\OUAZZE\OneDrive\Bureau\SupplyAI\supply_data.csv"
    
    # Allow override via command line
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    # Verify Ollama
    if not check_ollama():
        sys.exit(1)
    
    # Build and run
    try:
        rag_chain = build_system(csv_path)
        interactive_mode(rag_chain)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("\nPlease ensure your CSV file exists at the specified path.")
        print("Or provide a path: python main.py /path/to/your/data.csv")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        raise


if __name__ == "__main__":
    main()