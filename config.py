import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Base Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
INDEX_DIR = os.path.join(DATA_DIR, "llamaindex")

# Ensure base folders exist (safe on cloud)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# -----------------------------
# Embeddings
# -----------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------
# Retrieval
# -----------------------------
TOP_K = 5

# -----------------------------
# LLM / API
# -----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
