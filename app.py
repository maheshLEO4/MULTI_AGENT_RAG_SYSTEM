import os
import streamlit as st

from ingest import ingest_pdfs
from retriever import LlamaIndexHybridRetriever
from agents.workflow import AgentWorkflow
from config import UPLOAD_DIR, INDEX_DIR

# -------------------------
# Streamlit Page Setup
# -------------------------
st.set_page_config(page_title="Multi-Agentic RAG", layout="wide")
st.title("📚 Multi-Agentic RAG Chatbot")

# -------------------------
# Settings Sidebar
# -------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    enable_verification = st.checkbox(
        "Enable Verification", 
        value=False,
        help="🐌 Slower but validates answer accuracy. ⚡ Disable for 3x faster responses."
    )
    st.info(
        "⚡ **Fast Mode (Default)**: ~2-3 seconds\n\n"
        "🔍 **Verification Mode**: ~6-10 seconds but checks answer quality"
    )

# -------------------------
# Session State
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# -------------------------
# PDF Upload Section
# -------------------------
st.markdown("### 📄 Upload Documents")
st.info("💡 **Tip**: For large PDFs (>50MB or >200 pages), consider splitting them into smaller files for faster processing.")

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more PDF files. Large files will be processed in batches."
)

if uploaded_files:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Calculate total size
    total_size_mb = sum(file.size for file in uploaded_files) / (1024 * 1024)
    
    # Show warning for large files
    if total_size_mb > 50:
        st.warning(f"⚠️ Large upload detected ({total_size_mb:.1f} MB). Indexing may take 2-5 minutes.")
    
    # Save files
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        file_size_mb = file.size / (1024 * 1024)
        
        # Individual file size warning
        if file_size_mb > 100:
            st.warning(f"⚠️ {file.name} is very large ({file_size_mb:.1f} MB). Consider splitting it.")
        
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    # Progress bar for indexing
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(progress, message):
        progress_bar.progress(progress)
        status_text.text(message)
    
    try:
        ingest_pdfs(progress_callback=update_progress)
        
        # Reset retriever cache after new upload
        st.session_state.retriever = None
        
        progress_bar.empty()
        status_text.empty()
        st.success("✅ PDFs indexed successfully!")
    except MemoryError:
        progress_bar.empty()
        status_text.empty()
        st.error("❌ File too large! Try splitting the PDF into smaller parts (< 100 pages each).")
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error indexing PDFs: {str(e)}")

# -------------------------
# Chat History Display (SAFE)
# -------------------------
for msg in st.session_state.chat_history:
    if isinstance(msg, dict):
        # New format
        if "user" in msg and "assistant" in msg:
            st.chat_message("user").write(msg["user"])
            st.chat_message("assistant").write(msg["assistant"])
            
            # Display verification report if available
            if "verification" in msg and msg["verification"]:
                with st.expander("🔍 Verification Report", expanded=False):
                    st.markdown(msg["verification"])

        # Old / fallback format
        elif msg.get("role") == "user":
            st.chat_message("user").write(msg.get("content", ""))

        elif msg.get("role") == "assistant":
            st.chat_message("assistant").write(msg.get("content", ""))

# -------------------------
# Chat Input
# -------------------------
question = st.chat_input("Ask a question about the uploaded PDFs")

if question:
    # Guard: index must exist
    if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
        st.warning("⚠️ Please upload and index PDFs first.")
        st.stop()

    # Initialize retriever only once (cache in session state)
    if st.session_state.retriever is None:
        with st.spinner("Loading retriever..."):
            st.session_state.retriever = LlamaIndexHybridRetriever()
    
    retriever = st.session_state.retriever
    workflow = AgentWorkflow(enable_verification=enable_verification)

    # Show processing message
    with st.spinner("🤔 Thinking..." if not enable_verification else "🤔 Thinking and verifying..."):
        # 🔥 FIXED: Only pass question and retriever (2 arguments)
        result = workflow.full_pipeline(
            question,
            retriever
        )

    # Save chat in NEW SAFE FORMAT
    st.session_state.chat_history.append({
        "user": question,
        "assistant": result.get("draft_answer", ""),
        "verification": result.get("verification_report", "")
    })

    # Display current turn
    st.chat_message("user").write(question)
    st.chat_message("assistant").write(result.get("draft_answer", ""))
    
    # Display verification report if available
    verification_report = result.get("verification_report", "")
    if verification_report:
        with st.expander("🔍 Verification Report", expanded=False):
            st.markdown(verification_report)