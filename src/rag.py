from typing import Dict
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from src.config import OLLAMA_BASE_URL, LLM_MODEL
from src.retrieve import retrieve_chunks, format_context
from src.utils import format_sources


# Prompt that forces the LLM to only use retrieved context (no hallucination)
ANSWER_PROMPT = """You are a study assistant for an "Intro to Logic" course. Answer the question using ONLY the provided context from lecture slides.

Rules:
- Answer only using the context below
- If the context doesn't contain the answer, say "I don't have enough information from the slides to answer this question."
- Cite sources inline using the format [filename p.X]
- Be concise and exam-focused
- Do not add information not present in the context

Context:
{context}

Question: {question}

Answer:"""


def get_llm():
    """Initialize the Ollama LLM with low temperature for consistent answers."""
    return Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1  # low temp = more deterministic output
    )


def answer_question(question: str) -> Dict:
    """Main RAG flow: retrieve relevant chunks -> build prompt -> generate answer."""
    # Step 1: Find the most relevant chunks from the vector store
    chunks = retrieve_chunks(question)

    if not chunks:
        return {
            "answer": "I don't have enough information from the slides to answer this question.",
            "sources": [],
            "chunks": []
        }

    # Step 2: Format chunks into a context string with source citations
    context = format_context(chunks)

    # Step 3: Fill the prompt template with context + question
    prompt_template = PromptTemplate(
        template=ANSWER_PROMPT,
        input_variables=["context", "question"]
    )

    # Step 4: Send to LLM and get the answer
    llm = get_llm()
    prompt = prompt_template.format(context=context, question=question)
    answer = llm.invoke(prompt)

    sources = [{"source": c["source"], "page": c["page"]} for c in chunks]

    return {
        "answer": answer.strip(),
        "sources": sources,
        "chunks": chunks
    }
