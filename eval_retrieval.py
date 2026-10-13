import argparse
import json
import tempfile
from pathlib import Path

from langchain_chroma import Chroma

import config
from ingest import VectorStore


def run(docs_dir: Path, qa_path: Path, k: int, embeddings=None) -> dict:
    """
    Evaluate document retrieval using Hit Rate and Mean Reciprocal Rank (MRR).

    A temporary ChromaDB is created for evaluation. The project's existing
    production ChromaDB is not modified.
    """

    # Load evaluation questions
    qa = json.loads(qa_path.read_text(encoding="utf-8"))

    if not qa:
        raise ValueError(f"No evaluation questions found in {qa_path}")

    if k < 1:
        raise ValueError("k must be at least 1.")

    if not docs_dir.is_dir():
        raise FileNotFoundError(
            f"Documents directory not found: {docs_dir.resolve()}"
        )

    # Find supported documents
    files = sorted(
        file_path
        for file_path in docs_dir.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in config.ALLOWED_EXTENSIONS
    )

    if not files:
        raise ValueError(
            f"No supported documents found in {docs_dir.resolve()}. "
            f"Allowed extensions: {config.ALLOWED_EXTENSIONS}"
        )

    # Build the retrieval database inside a temporary directory
    with tempfile.TemporaryDirectory(prefix="docubot_eval_") as tmp:
        # Use the existing project's ingestion logic for loading and chunking.
        # The exact constructor arguments depend on your ingest.py implementation.
        helper = VectorStore(
            path=files[0],
            Chroma_db_dir=Path(tmp),
        )

        all_chunks = []

        for file_path in files:
            helper.path = file_path

            documents = helper.load_file()
            chunks = helper.chunk_pages(documents)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("Document ingestion produced no chunks.")

        # Reuse the project's embedding model unless one was supplied.
        embedding_model = embeddings or helper.embeddings

        # Create a temporary vector store without calling create_vectorstore(),
        # which may delete an existing database directory.
        store = Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding_model,
            collection_name="docubot_eval",
        )

        hits = 0
        reciprocal_rank_sum = 0.0
        misses = []

        for item in qa:
            question = item["question"]
            expected_phrase = item["expected_phrase"].strip().lower()

            results = store.similarity_search(question, k=k)

            rank = next(
                (
                    index
                    for index, document in enumerate(results, start=1)
                    if expected_phrase
                    in document.page_content.lower()
                ),
                None,
            )

            if rank is not None:
                hits += 1
                reciprocal_rank_sum += 1.0 / rank
            else:
                misses.append(question)

        total_questions = len(qa)

        return {
            "questions": total_questions,
            "k": k,
            "hit_rate": round(hits / total_questions, 3),
            "mrr": round(reciprocal_rank_sum / total_questions, 3),
            "misses": misses,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate DocuBot document retrieval."
    )

    parser.add_argument(
        "--docs",
        type=Path,
        default=Path("data"),
        help="Directory containing documents for evaluation.",
    )

    parser.add_argument(
        "--qa",
        type=Path,
        default=Path("data/qa_pairs.json"),
        help="Path to the evaluation question JSON file.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=config.TOP_K,
        help="Number of documents retrieved per question.",
    )

    args = parser.parse_args()

    try:
        result = run(
            docs_dir=args.docs,
            qa_path=args.qa,
            k=args.k,
        )
        print(json.dumps(result, indent=2))

    except Exception as error:
        parser.exit(1, f"Evaluation failed: {error}\n")