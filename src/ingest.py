import os
import json
from pathlib import Path
from datetime import datetime
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import PDF_DIR, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL, OLLAMA_BASE_URL
from src.utils import generate_chunk_id


def load_pdfs() -> List:
    documents = []
    pdf_dir = Path(PDF_DIR)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIR}")

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {PDF_DIR}")

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = pdf_path.name
            doc.metadata["page"] = doc.metadata.get("page", 0) + 1

        documents.extend(docs)

    return documents


def chunk_documents(documents: List) -> List:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["chunk_id"] = generate_chunk_id(
            chunk.page_content,
            chunk.metadata
        )

    return chunks


def build_index(chunks: List) -> FAISS:
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_index(vectorstore: FAISS):
    index_path = Path(INDEX_DIR)
    index_path.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(index_path))

    metadata = {
        "last_build": datetime.now().isoformat(),
        "num_chunks": vectorstore.index.ntotal,
        "embed_model": EMBED_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP
    }

    with open(index_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_index() -> FAISS:
    index_path = Path(INDEX_DIR)

    if not index_path.exists() or not (index_path / "index.faiss").exists():
        raise FileNotFoundError(f"Index not found at {INDEX_DIR}. Run ingestion first.")

    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    vectorstore = FAISS.load_local(
        str(index_path),
        embeddings
    )

    return vectorstore


def get_index_metadata() -> dict:
    metadata_path = Path(INDEX_DIR) / "metadata.json"

    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r") as f:
        return json.load(f)


def ingest_pipeline():
    documents = load_pdfs()
    chunks = chunk_documents(documents)
    vectorstore = build_index(chunks)
    save_index(vectorstore)

    return {
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "index_path": INDEX_DIR
    }


if __name__ == "__main__":
    from src.utils import verify_environment

    env_check = verify_environment()

    if not env_check["ollama_running"]:
        print("Error: Ollama is not running")
        exit(1)

    if not env_check["embed_exists"]:
        print(f"Error: Embedding model {EMBED_MODEL} not found")
        exit(1)

    result = ingest_pipeline()
    print(f"Ingestion complete: {result['num_documents']} PDFs, {result['num_chunks']} chunks")
