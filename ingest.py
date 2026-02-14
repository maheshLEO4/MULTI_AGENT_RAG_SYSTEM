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
    if not os.path.exists(UPLOAD_DIR):
        return

    docs = SimpleDirectoryReader(
        UPLOAD_DIR,
        required_exts=[".pdf"]
    ).load_data()

    if not docs:
        return

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(docs)

    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)
    keyword_index = KeywordTableIndex(nodes)

    storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    vector_index.storage_context.persist(persist_dir=INDEX_DIR)
    keyword_index.storage_context.persist(persist_dir=INDEX_DIR)

    print(f"✅ Ingested {len(nodes)} nodes from PDFs")
