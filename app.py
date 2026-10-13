import os
import tempfile
from pathlib import Path
import streamlit as st
from ingest import VectorStore
from rag import setup_rag_chain, ask_question

# ---------------- Configuration ----------------

st.set_page_config(
    page_title="DocuBot - AI Document Q&A",
    page_icon="📚",
    layout="wide",
)

st.title("📚 DocuBot")
st.caption("AI-powered document question-answering using RAG")

# Check API key
if not os.getenv("GROQ_API_KEY"):
    st.warning(
        "GROQ_API_KEY is not configured. "
        "Add it to your .env file before running the application."
    )
    st.stop()


# ---------------- Session State ----------------

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# ---------------- Sidebar ----------------

with st.sidebar:
    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"],
        help="Upload a document to create a searchable knowledge base.",
    )

    if uploaded_file is not None:
        st.info(f"Selected: {uploaded_file.name}")

    build_database = st.button(
        "Upload and Build",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    if build_database and uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()

        # Save upload temporarily so ingest.py can process it.
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=suffix,
                delete=False,
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            with st.spinner("Processing document and building vector database..."):
                try:
                    # This function must accept the uploaded file path.
                    vector_store = VectorStore(Path(temp_path))
                    vector_store.create_vectorstore()

                    # Initialize the RAG chain after ingestion.
                    st.session_state.rag_chain = setup_rag_chain()

                    st.session_state.document_name = uploaded_file.name
                    st.session_state.messages = []

                    st.success("Document processed successfully!")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

        except Exception as exc:
            st.session_state.rag_chain = None
            st.error(f"Could not process the document: {exc}")

    st.divider()

    if st.session_state.document_name:
        st.success(f"Active document: {st.session_state.document_name}")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Embeddings run locally. Answer generation uses the Groq API.")


# ---------------- Main Chat Interface ----------------

if st.session_state.rag_chain is None:
    st.info(
        "👈 Upload a PDF or TXT file and click "
        "**Build / Rebuild Database** to get started."
    )
else:
    st.subheader("💬 Ask Questions")

    # Display conversation history.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input(
        "Ask a question about your document..."
    )

    if user_question:
        # Show and store the user's question.
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate the answer.
        with st.chat_message("assistant"):
            with st.spinner("Searching your document and generating an answer..."):
                try:
                    answer = ask_question(
                        user_question.strip(),
                        st.session_state.rag_chain,
                    )

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": str(answer)}
                    )

                except Exception as exc:
                    st.error(f"Could not generate an answer: {exc}")