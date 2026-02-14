import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import UPLOAD_DIR, INDEX_DIR, EMBED_MODEL


def ingest_pdfs():
    """
    Ingest PDF documents from UPLOAD_DIR and create a vector store index.
    """
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(INDEX_DIR, exist_ok=True)

        # Disable OpenAI completely
        Settings.llm = None
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL
        )

        # Load documents
        docs = SimpleDirectoryReader(
            UPLOAD_DIR,
            required_exts=[".pdf"]
        ).load_data()

        if not docs:
            print("No PDF documents found in upload directory.")
            return

        print(f"Loaded {len(docs)} documents.")

        # Split documents into chunks
        splitter = SentenceSplitter(
            chunk_size=256,  # Reduced from 512 for faster processing
            chunk_overlap=25  # Reduced from 50
        )

        nodes = splitter.get_nodes_from_documents(docs)
        print(f"Created {len(nodes)} chunks.")

        # Create index
        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=INDEX_DIR)
        
        print(f"Index successfully created and persisted to {INDEX_DIR}")
    
    except Exception as e:
        print(f"Error during PDF ingestion: {e}")
        raise