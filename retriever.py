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
        if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
            raise RuntimeError("No index found. Upload PDFs first.")

        Settings.llm = None
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL
        )

        storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        index = load_index_from_storage(storage)

        self.vector = VectorIndexRetriever(
            index=index,
            similarity_top_k=TOP_K
        )

        self.bm25 = BM25Retriever.from_defaults(
            index=index,
            similarity_top_k=TOP_K
        )

    def invoke(self, query: str):
        vector_nodes = self.vector.retrieve(query)
        bm25_nodes = self.bm25.retrieve(query)

        seen = set()
        merged = []

        for n in vector_nodes + bm25_nodes:
            node_id = n.node.node_id
            if node_id not in seen:
                seen.add(node_id)
                merged.append(n)

        return [
            Document(
                page_content=n.node.text,
                metadata=n.node.metadata
            )
            for n in merged[:TOP_K]
        ]
