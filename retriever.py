import os
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers.fusion_retriever import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_core.documents import Document
from config import INDEX_DIR, TOP_K, EMBED_MODEL


class LlamaIndexHybridRetriever:
    def __init__(self):
        if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
            raise RuntimeError("No index found. Upload PDFs first.")

        # 🔒 Disable LLM globally (THIS FIXES THE ERROR)
        Settings.llm = None

        # 🔒 Ensure same embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL
        )

        storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage)

        vector = VectorIndexRetriever(
            index=index,
            similarity_top_k=TOP_K
        )

        bm25 = BM25Retriever.from_defaults(
            index=index,
            similarity_top_k=TOP_K
        )

        self.retriever = QueryFusionRetriever(
            retrievers=[vector, bm25],
            similarity_top_k=TOP_K,
            num_queries=1
        )

    def invoke(self, query: str):
        nodes = self.retriever.retrieve(query)

        return [
            Document(
                page_content=n.node.text,
                metadata=n.node.metadata
            )
            for n in nodes
        ]
