
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import GROQ_API_KEY, GROQ_MODEL
from ingest import VectorStore, CHROMA_DB_DIR


def format_docs(docs):
    """Convert retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def setup_rag_chain():
    """Initialize the RAG pipeline."""

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured.")

    # Load the existing vector database
    vector_store = VectorStore(
        path=None,
        Chroma_db_dir=CHROMA_DB_DIR
    )
    vector_store.load_vectorstore()

    # Use the underlying LangChain Chroma instance
    retriever = vector_store.vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )

    # Initialize the language model
    llm = ChatGroq(model=GROQ_MODEL)

    template ="""
    You are a helpful document question-answering assistant.

    Answer the question using only the supplied document context.
    If the answer is not present in the context, say that you
    could not find the answer in the document.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Build the RAG chain
    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(question: str, rag_chain=None) -> str:
    """Ask a question using the initialized RAG chain."""

    if rag_chain is None:
        raise ValueError(
            "RAG chain not initialized. Call setup_rag_chain() first."
        )

    return rag_chain.invoke(question)


if __name__ == "__main__":
    try:
        chain = setup_rag_chain()
        question = "Summarize the document in a few sentences."
        answer = ask_question(question, chain)

        print("Question:", question)
        print("Answer:", answer)

    except Exception as error:
        print(f"Execution failed: {error}")
