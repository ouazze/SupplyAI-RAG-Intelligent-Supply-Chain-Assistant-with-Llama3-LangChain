"""
SupplyAI Streamlit Application
Interactive UI for the Supply Chain AI Assistant.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from langchain_core.documents import Document

from data_loader.loader import SupplyChainDataLoader
from embeddings.embedder import SupplyChainEmbedder
from vectorstore.faiss_store import SupplyChainVectorStore
from llm.llama import SupplyChainLLM
from chains.rag_chain import SupplyChainRAGChain


# ─────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SupplyAI - Supply Chain Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Custom CSS for Professional Look
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
    .source-box {
        background-color: #fafafa;
        border-left: 3px solid #1f77b4;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 5px 5px 0;
    }
    .stChatMessage {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────────
def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "rag_chain": None,
        "data_loader": None,
        "chat_history": [],
        "inventory_summary": None,
        "system_ready": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─────────────────────────────────────────────────────────────
# System Initialization
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def initialize_system(csv_path: str):
    """
    Initialize the entire RAG pipeline.
    Cached to prevent re-initialization on every interaction.
    """
    try:
        # 1. Load Data
        loader = SupplyChainDataLoader(csv_path)
        documents = loader.to_documents()
        summary = loader.get_inventory_summary()
        
        # 2. Setup Embeddings & Vector Store
        embedder = SupplyChainEmbedder()
        vector_store = SupplyChainVectorStore(embedder=embedder)
        vector_store.get_or_create(documents)
        
        # 3. Setup LLM & RAG Chain
        llm = SupplyChainLLM()
        rag_chain = SupplyChainRAGChain(
            vector_store=vector_store,
            llm=llm,
            prompt_type="rag"
        )
        
        return rag_chain, loader, summary
        
    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        return None, None, None


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
def render_sidebar():
    """Render the application sidebar."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/supply-chain.png", width=80)
        st.title("SupplyAI")
        st.markdown("---")
        
        # Data Source Configuration
        st.subheader("📁 Data Source")
        csv_path = st.text_input(
            "CSV File Path",
            value=r"C:\Users\OUAZZE\OneDrive\Bureau\SupplyAI\supply_data.csv",
            help="Path to your supply_data.csv file"
        )
        
        # Model Configuration
        st.subheader("🤖 Model Settings")
        model_name = st.selectbox(
            "Ollama Model",
            options=["llama3", "llama3.1", "llama3:8b", "mistral"],
            index=0
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Lower = more deterministic, Higher = more creative"
        )
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        if st.session_state.system_ready:
            st.success("✅ System Ready")
            if st.session_state.inventory_summary:
                summary = st.session_state.inventory_summary
                st.metric("Total Products", summary["total_products"])
                st.metric("Low Stock Items", summary["low_stock_items"], delta=f"{summary['at_risk_items']} at risk")
        else:
            st.warning("⚠️ System Not Initialized")
        
        st.markdown("---")
        st.markdown("*Built with LangChain + FAISS + Ollama*")
        
        return csv_path, model_name, temperature


# ─────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────
def render_header():
    """Render the main header section."""
    st.markdown('<div class="main-header">🏭 SupplyAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Supply Chain Intelligence & Risk Management</div>', unsafe_allow_html=True)


def render_metrics():
    """Render inventory metrics cards."""
    if not st.session_state.inventory_summary:
        return
        
    summary = st.session_state.inventory_summary
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("📦 Total SKUs", summary["total_products"])
    with cols[1]:
        st.metric("🚨 Critical Stock", summary["low_stock_items"])
    with cols[2]:
        st.metric("⚠️ At Risk", summary["at_risk_items"])
    with cols[3]:
        healthy = summary["total_products"] - summary["at_risk_items"]
        st.metric("✅ Healthy", healthy)
    
    st.markdown("---")


def render_chat_interface():
    """Render the main chat interface."""
    st.subheader("💬 Ask SupplyAI")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📎 View Sources"):
                    for src in message["sources"]:
                        st.markdown(f"""
                        <div class="source-box">
                            <b>{src['product']}</b> (ID: {src['product_id']})<br>
                            📍 {src['warehouse']} | 🏭 {src['supplier']}<br>
                            📊 Stock: {src['stock']} units
                        </div>
                        """, unsafe_allow_html=True)
    
    # Input area
    if prompt := st.chat_input("Ask about inventory, risks, or recommendations..."):
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing supply chain data..."):
                try:
                    if st.session_state.rag_chain is None:
                        st.error("System not initialized. Please check your CSV path and click Initialize.")
                        return
                    
                    response = st.session_state.rag_chain.query(prompt)
                    answer = response["answer"]
                    sources = st.session_state.rag_chain.get_sources_metadata(response)
                    
                    st.markdown(answer)
                    
                    # Display sources in expander
                    if sources:
                        with st.expander("📎 Data Sources Used"):
                            for src in sources:
                                st.markdown(f"""
                                <div class="source-box">
                                    <b>{src['product']}</b> (ID: {src['product_id']})<br>
                                    📍 {src['warehouse']} | 🏭 {src['supplier']}<br>
                                    📊 Stock: {src['stock']} units
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def render_sample_questions():
    """Render sample question buttons."""
    st.markdown("---")
    st.subheader("🚀 Quick Actions")
    
    samples = [
        "Which products have low stock and need immediate restocking?",
        "What are the top 5 highest-risk items in our inventory?",
        "Analyze supplier performance and identify single-source risks.",
        "Generate a restock recommendation for warehouse A.",
        "Which items have lead times exceeding 30 days?"
    ]
    
    cols = st.columns(len(samples))
    for i, question in enumerate(samples):
        with cols[i]:
            if st.button(f"Q{i+1}", help=question, use_container_width=True):
                # Simulate chat input
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question
                })
                st.rerun()


# ─────────────────────────────────────────────────────────────
# Main Application Entry
# ─────────────────────────────────────────────────────────────
def main():
    init_session_state()
    csv_path, model_name, temperature = render_sidebar()
    render_header()
    
    # Initialize button
    if not st.session_state.system_ready:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Initialize SupplyAI System", use_container_width=True, type="primary"):
                with st.spinner("Loading data, building vector index, and connecting to Ollama..."):
                    rag_chain, loader, summary = initialize_system(csv_path)
                    if rag_chain:
                        st.session_state.rag_chain = rag_chain
                        st.session_state.data_loader = loader
                        st.session_state.inventory_summary = summary
                        st.session_state.system_ready = True
                        st.success("✅ SupplyAI is ready!")
                        st.rerun()
    
    else:
        render_metrics()
        render_chat_interface()
        render_sample_questions()


if __name__ == "__main__":
    main()