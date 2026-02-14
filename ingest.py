import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import UPLOAD_DIR, INDEX_DIR, EMBED_MODEL


def ingest_pdfs():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    # 🔒 Explicitly set embedding model (prevents OpenAI fallback)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL
    )

    docs = SimpleDirectoryReader(
        UPLOAD_DIR,
        required_exts=[".pdf"]
    ).load_data()

    if not docs:
        return

    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    nodes = splitter.get_nodes_from_documents(docs)

    vector_index = VectorStoreIndex(nodes)

    vector_index.storage_context.persist(persist_dir=INDEX_DIR)
