import os
import streamlit as st
from ingest import ingest_pdfs
from retriever import LlamaIndexHybridRetriever
from agents.workflow import AgentWorkflow
from config import UPLOAD_DIR, INDEX_DIR

st.set_page_config(page_title="Multi-Agentic RAG", layout="wide")
st.title("📚 Multi-Agentic RAG Chatbot")

# -------------------------
# Session State
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# Cached Retriever
# -------------------------
@st.cache_resource(show_spinner=False)
def get_retriever():
    return LlamaIndexHybridRetriever()

# -------------------------
# Upload PDFs
# -------------------------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
            f.write(file.getbuffer())

    with st.spinner("Indexing documents..."):
        ingest_pdfs()

    st.success("✅ PDFs indexed successfully")

# -------------------------
# Chat UI
# -------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the PDFs")

if question:
    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )

    if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
        st.warning("Please upload PDFs first.")
        st.stop()

    retriever = get_retriever()

    with st.spinner("Thinking..."):
        workflow = AgentWorkflow()
        result = workflow.full_pipeline(
            question=question,
            retriever=retriever,
            chat_history=st.session_state.chat_history
        )

    answer = result["draft_answer"]

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )

    st.rerun()
