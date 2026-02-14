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
# PDF Upload
# -------------------------
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    for file in uploaded_files:
        path = os.path.join(UPLOAD_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())

    with st.spinner("Indexing PDFs..."):
        ingest_pdfs()

    st.success("✅ PDFs indexed successfully!")

# -------------------------
# Chat History Display
# -------------------------
for msg in st.session_state.chat_history:
    st.chat_message("user").write(msg["user"])
    st.chat_message("assistant").write(msg["assistant"])

# -------------------------
# Chat Input
# -------------------------
question = st.chat_input("Ask a question about the uploaded PDFs")

if question:
    # Safety check
    if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
        st.warning("⚠️ Upload and index PDFs first.")
        st.stop()

    retriever = LlamaIndexHybridRetriever()
    workflow = AgentWorkflow()

    # 🔥 FIX: positional arguments ONLY
    result = workflow.full_pipeline(
        question,
        retriever,
        st.session_state.chat_history
    )

    # Save chat
    st.session_state.chat_history.append({
        "user": question,
        "assistant": result["draft_answer"]
    })

    # Display
    st.chat_message("user").write(question)
    st.chat_message("assistant").write(result["draft_answer"])
