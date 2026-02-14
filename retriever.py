import os
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
    QueryFusionRetriever
)
from langchain_core.documents import Document
from config import INDEX_DIR, TOP_K


class LlamaIndexHybridRetriever:
    def __init__(self):
        if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
            raise RuntimeError(
                "No index found. Please upload PDFs first."
            )

        storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)

        vector_index = load_index_from_storage(storage)
        keyword_index = load_index_from_storage(storage)

        vector = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=TOP_K
        )

        keyword = KeywordTableSimpleRetriever(index=keyword_index)

        self.retriever = QueryFusionRetriever(
            retrievers=[vector, keyword],
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
