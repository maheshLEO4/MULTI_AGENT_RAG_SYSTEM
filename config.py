import os

DATA_DIR = "data/docs"
CHROMA_DIR = "data/chroma"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 8

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
