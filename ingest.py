"""Load DarGlobal and Wasalt data, create chunks, generate embeddings,
and store vectors in FAISS.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
DATA_PATH = Path("data/combined_dataset.jsonl")
INDEX_PATH = Path("data/index")
TEMP_INDEX_PATH = Path("data/index_build")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

EMB_MODEL = "BAAI/bge-small-en-v1.5"

# Memory optimization
EMBEDDING_BATCH_SIZE = 8
EMBEDDING_THREADS = 2
EMBEDDING_PARALLEL = 1


def load_docs() -> list[Document]:
    """Load JSONL records and convert them into LangChain Documents."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    docs = []
    invalid_records = 0

    with DATA_PATH.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)

                if not isinstance(record, dict):
                    invalid_records += 1
                    continue

                text = str(record.get("text") or "").strip()

                if not text:
                    invalid_records += 1
                    continue

                metadata = {
                    "id": str(record.get("id", "")),
                    "url": str(record.get("source_url", "")),
                    "title": str(record.get("title", "")),
                    "company": str(record.get("company", "")),
                    "site": str(record.get("company", "")),
                    "category": str(record.get("category", "")),
                }

                docs.append(
                    Document(
                        page_content=text,
                        metadata=metadata,
                    )
                )

            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc

    print(f"Documents loaded: {len(docs)}")
    print(f"Invalid/empty records skipped: {invalid_records}")

    return docs


def create_chunks(docs: list[Document]) -> list[Document]:
    """Split documents into smaller overlapping text chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


def create_vector_store(chunks: list[Document]) -> FAISS:
    """Generate embeddings and create the FAISS vector index."""

    if not chunks:
        raise ValueError("No chunks available for indexing.")

    print(f"Loading embedding model: {EMB_MODEL}")
    print(f"Embedding batch size: {EMBEDDING_BATCH_SIZE}")
    print(f"CPU threads: {EMBEDDING_THREADS}")

    embeddings = FastEmbedEmbeddings(
        model_name=EMB_MODEL,
        batch_size=EMBEDDING_BATCH_SIZE,
        threads=EMBEDDING_THREADS,
        parallel=EMBEDDING_PARALLEL,
    )

    print("Generating embeddings and creating FAISS index...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


def verify_index(index_path: Path) -> None:
    """Verify that the saved FAISS index files exist and are not empty."""

    required_files = [
        index_path / "index.faiss",
        index_path / "index.pkl",
    ]

    for file_path in required_files:
        if not file_path.is_file() or file_path.stat().st_size == 0:
            raise RuntimeError(f"Index file missing or empty: {file_path}")


def save_vector_store(vector_store: FAISS) -> None:
    """Save and verify the new index, preserving the previous index as backup."""

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Remove any incomplete index from a previous interrupted run.
    if TEMP_INDEX_PATH.exists():
        shutil.rmtree(TEMP_INDEX_PATH)

    print("Saving new FAISS index to temporary folder...")

    vector_store.save_local(str(TEMP_INDEX_PATH))

    # Verify the newly generated index before touching the old one.
    verify_index(TEMP_INDEX_PATH)

    old_index_backup = None

    if INDEX_PATH.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        old_index_backup = INDEX_PATH.parent / f"index_backup_{timestamp}"

        print(f"Creating backup of existing index: {old_index_backup}")

        # Copy the existing index instead of renaming it.
        shutil.copytree(INDEX_PATH, old_index_backup)

        verify_index(old_index_backup)

    try:
        # Remove the previous index only after its backup is verified.
        if INDEX_PATH.exists():
            shutil.rmtree(INDEX_PATH)

        # shutil.move handles cross-device moves using copy + delete.
        shutil.move(str(TEMP_INDEX_PATH), str(INDEX_PATH))

        # Verify the installed index.
        verify_index(INDEX_PATH)

    except Exception:
        print("Index installation failed. Attempting rollback...")

        if INDEX_PATH.exists():
            shutil.rmtree(INDEX_PATH)

        if old_index_backup and old_index_backup.exists():
            shutil.copytree(old_index_backup, INDEX_PATH)
            verify_index(INDEX_PATH)
            print("Previous FAISS index restored.")

        raise

    print("New FAISS index installed successfully.")

    if old_index_backup:
        print(f"Previous index backup preserved at: {old_index_backup}")


def main() -> None:
    """Run the complete ingestion pipeline."""

    print("\nStarting DarGlobal x Wasalt ingestion...\n")

    docs = load_docs()

    if not docs:
        raise ValueError("No valid documents found.")

    chunks = create_chunks(docs)

    vector_store = create_vector_store(chunks)

    save_vector_store(vector_store)

    print("\nIngestion completed successfully!")
    print(f"Source documents: {len(docs)}")
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"FAISS index location: {INDEX_PATH}")


if __name__ == "__main__":
    main()
