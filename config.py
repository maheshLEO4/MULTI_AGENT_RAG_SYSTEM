import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
INDEX_DIR = os.path.join(DATA_DIR, "llamaindex")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3  # Reduced from 5 for faster retrieval

# Large file handling settings
BATCH_SIZE = 1000  # Process nodes in batches for large files
MAX_FILE_SIZE_MB = 100  # Warn for files larger than this
CHUNK_SIZE = 256  # Optimized for speed and large files
CHUNK_OVERLAP = 25  # Reduced overlap for efficiency

GROQ_API_KEY = os.getenv("GROQ_API_KEY")