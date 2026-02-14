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


class LlamaIndexHybridRetriever:
    def __init__(self):
        """
        Initialize the hybrid retriever with both vector and BM25 retrieval.
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
            storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
            index = load_index_from_storage(storage)
        except Exception as e:
            raise RuntimeError(f"Failed to load index: {e}")

        # Initialize retrievers
        try:
            self.vector = VectorIndexRetriever(
                index=index,
                similarity_top_k=TOP_K
            )

            self.bm25 = BM25Retriever.from_defaults(
                index=index,
                similarity_top_k=TOP_K
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize retrievers: {e}")

    def invoke(self, query: str):
        """
        Retrieve documents using hybrid approach (vector + BM25).
        
        Args:
            query: The search query
            
        Returns:
            List of LangChain Document objects
        """
        try:
            # Retrieve from both retrievers
            vector_nodes = self.vector.retrieve(query)
            bm25_nodes = self.bm25.retrieve(query)
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

        # Merge results and deduplicate
        seen = set()
        merged = []

        for n in vector_nodes + bm25_nodes:
            node_id = n.node.node_id
            if node_id not in seen:
                seen.add(node_id)
                merged.append(n)

        # Convert to LangChain Documents
        return [
            Document(
                page_content=n.node.text,
                metadata=n.node.metadata or {}
            )
            for n in merged[:TOP_K]
        ]