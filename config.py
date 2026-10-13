import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

CHROMA_DB_DIR =  Path("data/chroma_db")
COLLECTION_NAME = "docubot"
# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-120b"

# Embedding model: keep consistent between indexing and retrieval
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Text splitting
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 150

if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be greater than zero.")

if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be >= 0 and less than CHUNK_SIZE."
    )

# Retrieval
RETRIEVER_K = 3
TOP_K = 4

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}