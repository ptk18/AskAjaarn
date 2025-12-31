import streamlit as st
from pathlib import Path
from src.utils import verify_environment, format_sources
from src.rag import answer_question
from src.study_modes import generate_quiz, generate_flashcards, export_flashcards_json
from src.ingest import get_index_metadata, ingest_pipeline


st.set_page_config(page_title="AskAjaan", layout="centered")

st.title("AskAjaan")
st.caption("Local RAG Study Bot for Intro to Logic")


def check_system_ready():
    env = verify_environment()

    if not env["ollama_running"]:
        st.error("Ollama is not running. Please start Ollama first.")
        st.stop()

    if not env["llm_exists"]:
        st.error("LLM model not found. Please install the required model.")
        st.stop()

    if not env["embed_exists"]:
        st.error("Embedding model not found. Please install the required model.")
        st.stop()

    metadata = get_index_metadata()
    if not metadata:
        st.warning("Index not built. Please build the index first using the Index Info tab.")
        return False

    return True


tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Quiz", "Flashcards", "Index Info"])

with tab1:
    st.subheader("Ask a Question")

    if check_system_ready():
        question = st.text_input("Question:", placeholder="What is modus ponens?")
        show_chunks = st.checkbox("Show retrieved chunks", value=False)

        if st.button("Ask", type="primary"):
            if question:
                with st.spinner("Searching slides..."):
                    result = answer_question(question)

                st.markdown("**Answer:**")
                st.write(result["answer"])

                if result["sources"]:
                    with st.expander("Sources"):
                        st.write(format_sources(result["sources"]))

                if show_chunks and result["chunks"]:
                    with st.expander("Retrieved Chunks"):
                        for i, chunk in enumerate(result["chunks"], 1):
                            st.markdown(f"**Chunk {i}** [{chunk['source']} p.{chunk['page']}]")
                            st.text(chunk["content"])
                            st.divider()
            else:
                st.warning("Please enter a question.")

with tab2:
    st.subheader("Generate Quiz")

    if check_system_ready():
        topic = st.text_input("Topic:", placeholder="propositional logic")
        num_questions = st.slider("Number of questions:", 3, 10, 5)

        if st.button("Generate Quiz", type="primary"):
            if topic:
                with st.spinner("Generating quiz..."):
                    result = generate_quiz(topic, num_questions)

                st.markdown("**Quiz:**")
                st.write(result["quiz"])

                if result["sources"]:
                    with st.expander("Sources"):
                        st.write(format_sources(result["sources"]))
            else:
                st.warning("Please enter a topic.")

with tab3:
    st.subheader("Generate Flashcards")

    if check_system_ready():
        topic = st.text_input("Topic:", placeholder="truth tables", key="flashcard_topic")

        if st.button("Generate Flashcards", type="primary"):
            if topic:
                with st.spinner("Generating flashcards..."):
                    result = generate_flashcards(topic)

                if result["flashcards"]:
                    st.markdown("**Flashcards:**")

                    for i, card in enumerate(result["flashcards"], 1):
                        with st.expander(f"Card {i}: {card.get('question', 'N/A')[:60]}..."):
                            st.markdown(f"**Q:** {card.get('question', 'N/A')}")
                            st.markdown(f"**A:** {card.get('answer', 'N/A')}")
                            if card.get('source'):
                                st.caption(f"Source: {card['source']}")

                    json_data = export_flashcards_json(result["flashcards"])
                    st.download_button(
                        label="Download as JSON",
                        data=json_data,
                        file_name="flashcards.json",
                        mime="application/json"
                    )
                else:
                    st.write(result["raw_text"])

                if result["sources"]:
                    with st.expander("Sources"):
                        st.write(format_sources(result["sources"]))
            else:
                st.warning("Please enter a topic.")

with tab4:
    st.subheader("Index Information")

    env = verify_environment()

    st.markdown("**System Status:**")
    st.write(f"Ollama running: {'Yes' if env['ollama_running'] else 'No'}")
    st.write(f"LLM model available: {'Yes' if env['llm_exists'] else 'No'}")
    st.write(f"Embedding model available: {'Yes' if env['embed_exists'] else 'No'}")

    metadata = get_index_metadata()

    if metadata:
        st.markdown("**Index Status:**")
        st.write(f"Total chunks: {metadata.get('num_chunks', 'N/A')}")
        st.write(f"Embedding model: {metadata.get('embed_model', 'N/A')}")
        st.write(f"Chunk size: {metadata.get('chunk_size', 'N/A')}")
        st.write(f"Chunk overlap: {metadata.get('chunk_overlap', 'N/A')}")
        st.write(f"Last build: {metadata.get('last_build', 'N/A')}")
    else:
        st.warning("Index not built yet.")

    st.divider()

    if st.button("Build/Rebuild Index"):
        if not env["ollama_running"] or not env["embed_exists"]:
            st.error("Cannot build index. Check system status above.")
        else:
            with st.spinner("Building index... This may take a few minutes."):
                try:
                    result = ingest_pipeline()
                    st.success(f"Index built successfully: {result['num_documents']} PDFs, {result['num_chunks']} chunks")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error building index: {str(e)}")
