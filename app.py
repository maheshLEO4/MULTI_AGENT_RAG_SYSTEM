import os
import streamlit as st
from agents.workflow import AgentWorkflow
from llamaindex_ingest import ingest_pdfs
from llamaindex_retriever import LlamaIndexHybridRetriever
from config import UPLOAD_DIR

st.set_page_config("Multi-Agentic RAG", layout="wide")

st.title("📚 Multi-Agentic RAG (PDF Upload)")

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
        with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
            f.write(file.getbuffer())

    with st.spinner("Indexing PDFs..."):
        ingest_pdfs()

    st.success("PDFs indexed successfully!")

# -------------------------
# Question Input
# -------------------------
question = st.text_input("Ask a question about the uploaded PDFs")

if st.button("Run RAG") and question:
    with st.spinner("Running multi-agent workflow..."):
        retriever = LlamaIndexHybridRetriever()
        workflow = AgentWorkflow()

        result = workflow.full_pipeline(
            question=question,
            retriever=retriever
        )

    st.subheader("🧠 Answer")
    st.write(result["draft_answer"])

    st.subheader("✅ Verification")
    st.markdown(result["verification_report"])
