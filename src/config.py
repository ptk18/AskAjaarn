import os
from pathlib import Path

# Project root directory (one level up from src/)
BASE_DIR = Path(__file__).parent.parent

# Where PDFs and FAISS index are stored
PDF_DIR = os.getenv("PDF_DIR", str(BASE_DIR / "data" / "pdfs"))
INDEX_DIR = os.getenv("INDEX_DIR", str(BASE_DIR / "data" / "faiss_index"))

# Ollama connection and model settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")           # for generating answers
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text") # for creating embeddings

# Chunking and retrieval settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))      # max characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150")) # overlap between chunks
TOP_K = int(os.getenv("TOP_K", "5"))                   # number of chunks to retrieve per query
