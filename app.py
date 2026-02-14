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
# Session State
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# PDF Upload Section
# -------------------------
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    with st.spinner("Indexing PDFs..."):
        ingest_pdfs()

    st.success("✅ PDFs indexed successfully!")

# -------------------------
# Chat History Display (SAFE)
# -------------------------
for msg in st.session_state.chat_history:
    if isinstance(msg, dict):
        # New format
        if "user" in msg and "assistant" in msg:
            st.chat_message("user").write(msg["user"])
            st.chat_message("assistant").write(msg["assistant"])

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

    retriever = LlamaIndexHybridRetriever()
    workflow = AgentWorkflow()

    # 🔥 FIXED: Only pass question and retriever (2 arguments)
    result = workflow.full_pipeline(
        question,
        retriever
    )

    # Save chat in NEW SAFE FORMAT
    st.session_state.chat_history.append({
        "user": question,
        "assistant": result.get("draft_answer", "")
    })

    # Display current turn
    st.chat_message("user").write(question)
    st.chat_message("assistant").write(result.get("draft_answer", ""))