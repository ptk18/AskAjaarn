import json
from typing import Dict, List
from langchain.prompts import PromptTemplate
from src.rag import get_llm
from src.retrieve import retrieve_chunks, format_context


QUIZ_PROMPT = """You are creating an exam for an "Intro to Logic" course. Generate {num_questions} exam-style questions based ONLY on the provided context.

Rules:
- Questions must be answerable from the context
- Include a mix of definitions, applications, and conceptual questions
- Provide an answer key with explanations
- Cite slide sources for each question
- Format as numbered questions followed by answer key

Context:
{context}

Topic: {topic}

Generate {num_questions} questions:"""


FLASHCARD_PROMPT = """You are creating study flashcards for an "Intro to Logic" course. Generate flashcards based ONLY on the provided context.

Rules:
- Create clear Q/A pairs
- Focus on definitions, key concepts, and important distinctions
- Keep questions specific and answers concise
- Cite the slide source for each card
- Generate 8-10 flashcards

Context:
{context}

Topic: {topic}

Generate flashcards in this format:
Q: [question]
A: [answer]
Source: [citation]
---"""


def generate_quiz(topic: str, num_questions: int = 5) -> Dict:
    chunks = retrieve_chunks(topic, k=8)

    if not chunks:
        return {
            "quiz": "Not enough material found on this topic in the slides.",
            "sources": []
        }

    context = format_context(chunks)

    prompt_template = PromptTemplate(
        template=QUIZ_PROMPT,
        input_variables=["context", "topic", "num_questions"]
    )

    llm = get_llm()
    prompt = prompt_template.format(
        context=context,
        topic=topic,
        num_questions=num_questions
    )
    quiz = llm.invoke(prompt)

    sources = [{"source": c["source"], "page": c["page"]} for c in chunks]

    return {
        "quiz": quiz.strip(),
        "sources": sources
    }


def generate_flashcards(topic: str) -> Dict:
    chunks = retrieve_chunks(topic, k=8)

    if not chunks:
        return {
            "flashcards": [],
            "raw_text": "Not enough material found on this topic in the slides.",
            "sources": []
        }

    context = format_context(chunks)

    prompt_template = PromptTemplate(
        template=FLASHCARD_PROMPT,
        input_variables=["context", "topic"]
    )

    llm = get_llm()
    prompt = prompt_template.format(context=context, topic=topic)
    flashcards_text = llm.invoke(prompt)

    flashcards = parse_flashcards(flashcards_text)
    sources = [{"source": c["source"], "page": c["page"]} for c in chunks]

    return {
        "flashcards": flashcards,
        "raw_text": flashcards_text.strip(),
        "sources": sources
    }


def parse_flashcards(text: str) -> List[Dict]:
    cards = []
    current_card = {}

    for line in text.split("\n"):
        line = line.strip()

        if line.startswith("Q:"):
            if current_card:
                cards.append(current_card)
            current_card = {"question": line[2:].strip()}

        elif line.startswith("A:") and current_card:
            current_card["answer"] = line[2:].strip()

        elif line.startswith("Source:") and current_card:
            current_card["source"] = line[7:].strip()

        elif line == "---" and current_card:
            cards.append(current_card)
            current_card = {}

    if current_card and "question" in current_card:
        cards.append(current_card)

    return cards


def export_flashcards_json(flashcards: List[Dict]) -> str:
    return json.dumps(flashcards, indent=2)
