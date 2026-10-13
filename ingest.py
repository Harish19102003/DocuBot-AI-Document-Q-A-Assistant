
import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_DB_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME


def embedding(model_name=EMBEDDING_MODEL):
    """Create the embedding model used by indexing and retrieval."""
    return HuggingFaceEmbeddings(model_name=model_name)


class VectorStore:
    def __init__(
        self,
        path,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        collection_name=COLLECTION_NAME,
        Chroma_db_dir=CHROMA_DB_DIR,
        embeddings=None,
    ):
        self.path = path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.chroma_db_dir = Chroma_db_dir
        self.embeddings = embedding()
        self.vectordb = None

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(
                "chunk_overlap must be >= 0 and less than chunk_size."
            )

    def load_file(self):
        """Load a PDF or text file into LangChain documents."""
        if self.path is None:
            raise ValueError("A file path is required for ingestion.")

        if not self.path.is_file():
            raise FileNotFoundError(f"File not found: {self.path}")

        suffix = self.path.suffix.lower()

        if suffix == ".pdf":
            documents = PyPDFLoader(str(self.path)).load()
        elif suffix in {".txt", ".md"}:
            documents = TextLoader(str(self.path)).load()
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                "Use PDF, TXT, or MD."
            )

        if not documents or not any(
            doc.page_content.strip() for doc in documents
        ):
            raise ValueError(
                "No readable text found. "
                "Scanned PDFs may require OCR."
            )

        return documents

    def chunk_pages(self, documents):
        """Split the supplied documents without loading them again."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("Document produced no text chunks.")

        print(f"Split into {len(chunks)} chunks.")
        return chunks

    def create_vectorstore(self):
        """Ingest a document and create a persistent Chroma database."""
        documents = self.load_file()
        chunks = self.chunk_pages(documents)
        # Clean up previous database if it exists
        if os.path.exists(self.chroma_db_dir):
            shutil.rmtree(self.chroma_db_dir)
            print("Cleaned up existing ChromaDB directory.")
        Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.chroma_db_dir),
        )
        return self.load_vectorstore()

    def load_vectorstore(self):
        """Open the existing persistent Chroma database."""
        if not self.chroma_db_dir.is_dir():
            raise FileNotFoundError(
                f"Vector database not found: {self.chroma_db_dir}. "
                "Ingest a document first."
            )

        db_file = self.chroma_db_dir / "chroma.sqlite3"
        if not db_file.is_file():
            raise FileNotFoundError(
                f"Chroma database file missing: {db_file}"
            )

        self.vectordb = Chroma(
            collection_name=self.collection_name,
            persist_directory=str(self.chroma_db_dir),
            embedding_function=self.embeddings,
        )

        count = self.vectordb._collection.count()
        if count == 0:
            raise ValueError(
                "The Chroma collection is empty. Ingest a document first."
            )

        print(f"Loaded ChromaDB: {count} chunks.")
        return self.vectordb

    def query(self, query_text, top_k=5):
        """Find the most similar document chunks."""
        if self.vectordb is None:
            raise ValueError("Call load_vectorstore() first.")

        if not query_text.strip():
            return []

        return self.vectordb.similarity_search(
            query_text,
            k=top_k,
        )

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ingest.py path/to/document.pdf")
        raise SystemExit(1)

    file_path = Path(sys.argv[1]).resolve()

    try:
        store = VectorStore(path=file_path)
        store.create_vectorstore()
        print("Ingestion completed successfully.")
    except Exception as exc:
        print(f"Ingestion failed: {exc}")
        raise SystemExit(1)