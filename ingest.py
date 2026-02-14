import os
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import UPLOAD_DIR, INDEX_DIR, EMBED_MODEL, BATCH_SIZE, CHUNK_SIZE, CHUNK_OVERLAP
import streamlit as st


def ingest_pdfs(progress_callback=None):
    """
    Ingest PDF documents from UPLOAD_DIR and create a vector store index.
    Optimized for large files with batching and progress tracking.
    
    Args:
        progress_callback: Optional callback function to report progress
    """
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(INDEX_DIR, exist_ok=True)

        # Disable OpenAI completely
        Settings.llm = None
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL
        )

        if progress_callback:
            progress_callback(0.1, "Loading PDF documents...")

        # Load documents with file size limit handling
        docs = SimpleDirectoryReader(
            UPLOAD_DIR,
            required_exts=[".pdf"],
            filename_as_id=True  # Use filename as ID for tracking
        ).load_data()

        if not docs:
            print("No PDF documents found in upload directory.")
            return

        print(f"Loaded {len(docs)} documents.")
        
        if progress_callback:
            progress_callback(0.3, f"Loaded {len(docs)} documents. Splitting into chunks...")

        # Split documents into chunks
        splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,  # From config
            chunk_overlap=CHUNK_OVERLAP  # From config
        )

        nodes = splitter.get_nodes_from_documents(docs)
        total_nodes = len(nodes)
        print(f"Created {total_nodes} chunks.")

        if progress_callback:
            progress_callback(0.5, f"Created {total_nodes} chunks. Building vector index...")

        # For large files, process in batches
        BATCH_SIZE = 1000  # Process 1000 nodes at a time
        
        if total_nodes <= BATCH_SIZE:
            # Small dataset - process all at once
            index = VectorStoreIndex(nodes)
            if progress_callback:
                progress_callback(0.9, "Persisting index to disk...")
        else:
            # Large dataset - process in batches
            print(f"Large dataset detected ({total_nodes} nodes). Processing in batches...")
            
            # Create index with first batch
            first_batch = nodes[:BATCH_SIZE]
            index = VectorStoreIndex(first_batch)
            
            if progress_callback:
                progress_callback(0.6, f"Processed {BATCH_SIZE}/{total_nodes} chunks...")
            
            # Add remaining nodes in batches
            for i in range(BATCH_SIZE, total_nodes, BATCH_SIZE):
                batch = nodes[i:i + BATCH_SIZE]
                for node in batch:
                    index.insert_nodes([node])
                
                progress = 0.6 + (0.3 * (i / total_nodes))
                if progress_callback:
                    progress_callback(
                        progress, 
                        f"Processed {min(i + BATCH_SIZE, total_nodes)}/{total_nodes} chunks..."
                    )
                
                print(f"Processed {min(i + BATCH_SIZE, total_nodes)}/{total_nodes} chunks")

        # Persist index
        index.storage_context.persist(persist_dir=INDEX_DIR)
        
        if progress_callback:
            progress_callback(1.0, "✅ Indexing complete!")
        
        print(f"Index successfully created and persisted to {INDEX_DIR}")
        print(f"Total chunks indexed: {total_nodes}")
    
    except MemoryError as e:
        error_msg = "❌ Out of memory! File is too large. Try splitting the PDF into smaller files."
        print(error_msg)
        if progress_callback:
            progress_callback(1.0, error_msg)
        raise MemoryError(error_msg) from e
    
    except Exception as e:
        error_msg = f"Error during PDF ingestion: {e}"
        print(error_msg)
        if progress_callback:
            progress_callback(1.0, f"❌ {error_msg}")
        raise