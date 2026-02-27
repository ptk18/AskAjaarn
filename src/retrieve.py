from typing import List, Dict
from src.config import TOP_K
from src.ingest import load_index


def retrieve_chunks(query: str, k: int = TOP_K) -> List[Dict]:
    """Search the FAISS index for the top-k most similar chunks to the query."""
    vectorstore = load_index()

    # FAISS returns (document, similarity_score) pairs
    results = vectorstore.similarity_search_with_score(query, k=k)

    chunks = []
    for doc, score in results:
        chunks.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "?"),
            "score": float(score)  # lower score = more similar in FAISS
        })

    return chunks


def format_context(chunks: List[Dict]) -> str:
    """Turn retrieved chunks into a numbered context string with source labels."""
    if not chunks:
        return ""

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source_ref = f"[{chunk['source']} p.{chunk['page']}]"
        context_parts.append(f"Source {i} {source_ref}:\n{chunk['content']}")

    return "\n\n".join(context_parts)
