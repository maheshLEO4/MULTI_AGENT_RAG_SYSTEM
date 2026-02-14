import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.keyword_table import KeywordTableIndex
from config import UPLOAD_DIR, INDEX_DIR, EMBED_MODEL


def ingest_pdfs():
    # 🔒 Ensure directories exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Load PDFs
    docs = SimpleDirectoryReader(
        UPLOAD_DIR,
        required_exts=[".pdf"]
    ).load_data()

    if not docs:
        print("⚠️ No PDFs found for ingestion.")
        return

    # Split into chunks
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    nodes = splitter.get_nodes_from_documents(docs)

    # Embeddings
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # Build indexes
    vector_index = VectorStoreIndex(
        nodes,
        embed_model=embed_model
    )
    keyword_index = KeywordTableIndex(nodes)

    # Persist indexes
    storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    vector_index.storage_context.persist(persist_dir=INDEX_DIR)
    keyword_index.storage_context.persist(persist_dir=INDEX_DIR)

    print(f"✅ Ingested {len(nodes)} nodes from PDFs")
