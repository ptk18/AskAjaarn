import hashlib
import requests
from typing import Optional
from src.config import OLLAMA_BASE_URL, LLM_MODEL, EMBED_MODEL


def check_ollama_running() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def check_model_exists(model_name: str) -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return any(model["name"].startswith(model_name) for model in models)
        return False
    except Exception:
        return False


def verify_environment() -> dict:
    results = {
        "ollama_running": check_ollama_running(),
        "llm_exists": False,
        "embed_exists": False
    }

    if results["ollama_running"]:
        results["llm_exists"] = check_model_exists(LLM_MODEL)
        results["embed_exists"] = check_model_exists(EMBED_MODEL)

    return results


def generate_chunk_id(text: str, metadata: dict) -> str:
    content = f"{metadata.get('source', '')}{metadata.get('page', '')}{text}"
    return hashlib.md5(content.encode()).hexdigest()


def format_sources(sources: list) -> str:
    unique_sources = {}
    for src in sources:
        key = (src.get("source", ""), src.get("page", ""))
        if key not in unique_sources:
            unique_sources[key] = src

    formatted = []
    for src in unique_sources.values():
        source_name = src.get("source", "Unknown")
        page = src.get("page", "?")
        formatted.append(f"[{source_name} p.{page}]")

    return ", ".join(formatted)
