import os
import streamlit as st
from agents.workflow import AgentWorkflow
from ingest import ingest_pdfs
from retriever import LlamaIndexHybridRetriever
from config import UPLOAD_DIR, INDEX_DIR

st.set_page_config(page_title="Multi-Agentic RAG", layout="wide")
st.title("📚 Multi-Agentic RAG (PDF Upload)")

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
# Question Input
# -------------------------
question = st.text_input("Ask a question about the uploaded PDFs")

if st.button("Run RAG") and question:
    # 🔒 Guard: ensure index exists
    if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
        st.warning("⚠️ Please upload and index PDFs before asking questions.")
        st.stop()

    try:
        retriever = LlamaIndexHybridRetriever()
    except Exception as e:
        st.error(f"Retriever error: {e}")
        st.stop()

    with st.spinner("Running multi-agent workflow..."):
        workflow = AgentWorkflow()
        result = workflow.full_pipeline(
            question=question,
            retriever=retriever
        )

    st.subheader("🧠 Answer")
    st.write(result.get("draft_answer", "No answer generated."))

    st.subheader("✅ Verification Report")
    st.markdown(result.get("verification_report", "No verification report."))
