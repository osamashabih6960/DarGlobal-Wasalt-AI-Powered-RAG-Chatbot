
"""Load DarGlobal and Wasalt data, create chunks, generate embeddings,
and store vectors in FAISS.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

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
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

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


def save_vector_store(vector_store: FAISS) -> None:
    """Save the new index safely before replacing the existing index."""

    if TEMP_INDEX_PATH.exists():
        shutil.rmtree(TEMP_INDEX_PATH)

    print("Saving new FAISS index to temporary folder...")

    vector_store.save_local(str(TEMP_INDEX_PATH))

    # Verify both FAISS files exist before replacing the old index.
    required_files = [
        TEMP_INDEX_PATH / "index.faiss",
        TEMP_INDEX_PATH / "index.pkl",
    ]

    for file_path in required_files:
        if not file_path.exists() or file_path.stat().st_size == 0:
            raise RuntimeError(f"Index file missing or empty: {file_path}")

    old_index_backup = None

    if INDEX_PATH.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        old_index_backup = Path(f"data/index_replaced_{timestamp}")

        INDEX_PATH.rename(old_index_backup)

    try:
        TEMP_INDEX_PATH.rename(INDEX_PATH)

    except Exception:
        # Restore the old index if replacement fails.
        if old_index_backup and old_index_backup.exists():
            old_index_backup.rename(INDEX_PATH)
        raise

    print("New FAISS index installed successfully.")

    if old_index_backup:
        print(f"Previous index preserved at: {old_index_backup}")


def main() -> None:
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
