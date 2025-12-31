import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PDF_DIR = os.getenv("PDF_DIR", str(BASE_DIR / "data" / "pdfs"))
INDEX_DIR = os.getenv("INDEX_DIR", str(BASE_DIR / "data" / "faiss_index"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
