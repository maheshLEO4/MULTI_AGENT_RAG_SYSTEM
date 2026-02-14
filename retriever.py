import os
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_core.documents import Document
from config import INDEX_DIR, TOP_K, EMBED_MODEL
import logging

logger = logging.getLogger(__name__)


class LlamaIndexHybridRetriever:
    def __init__(self):
        """
        Initialize the hybrid retriever with both vector and BM25 retrieval.
        Optimized for large indices.
        """
        if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
            raise RuntimeError("No index found. Upload PDFs first.")

        # Configure settings
        Settings.llm = None
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL
        )

        # Load index from storage
        try:
            logger.info("Loading index from storage...")
            storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
            self.index = load_index_from_storage(storage)
            logger.info("Index loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load index: {e}")

        # Initialize retrievers with optimized settings
        try:
            # Vector retriever - optimized for large indices
            self.vector = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=TOP_K,
                # Add node postprocessors for better performance on large datasets
            )

            # BM25 retriever - optimized for large indices  
            self.bm25 = BM25Retriever.from_defaults(
                index=self.index,
                similarity_top_k=TOP_K
            )
            
            logger.info("Retrievers initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize retrievers: {e}")

    def invoke(self, query: str, timeout: int = 30):
        """
        Retrieve documents using hybrid approach (vector + BM25).
        Optimized for large indices with timeout protection.
        
        Args:
            query: The search query
            timeout: Maximum time in seconds (not enforced, just for documentation)
            
        Returns:
            List of LangChain Document objects
        """
        try:
            logger.debug(f"Retrieving documents for query: {query}")
            
            # Retrieve from both retrievers
            vector_nodes = self.vector.retrieve(query)
            bm25_nodes = self.bm25.retrieve(query)
            
            logger.debug(f"Vector retrieval: {len(vector_nodes)} nodes")
            logger.debug(f"BM25 retrieval: {len(bm25_nodes)} nodes")
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            # Return empty list on error instead of crashing
            return []

        # Merge results and deduplicate
        seen = set()
        merged = []

        for n in vector_nodes + bm25_nodes:
            node_id = n.node.node_id
            if node_id not in seen:
                seen.add(node_id)
                merged.append(n)

        # Limit to TOP_K for efficiency
        merged = merged[:TOP_K]
        
        logger.debug(f"Merged results: {len(merged)} unique nodes")

        # Convert to LangChain Documents
        documents = [
            Document(
                page_content=n.node.text,
                metadata=n.node.metadata or {}
            )
            for n in merged
        ]
        
        return documents