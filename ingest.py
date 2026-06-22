import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

DOCUMENTS_DIR = Path(__file__).parent / "documents"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "company_policies"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def extract_text_from_file(file_path: Path) -> str:
    if file_path.suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_path.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks


def ingest():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except (ValueError, chromadb.errors.NotFoundError):
        pass

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-base-en-v1.5")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_fn,
    )

    all_chunks = []
    all_ids = []
    all_metadata = []

    for file_path in sorted(DOCUMENTS_DIR.iterdir()):
        if file_path.suffix not in (".txt", ".pdf"):
            continue
        print(f"Processing: {file_path.name}")
        text = extract_text_from_file(file_path)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{file_path.stem}_{i}")
            all_metadata.append({"source": file_path.name})

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadata)
    print(f"Ingested {len(all_chunks)} chunks from {len(list(DOCUMENTS_DIR.iterdir()))} files.")


if __name__ == "__main__":
    ingest()
