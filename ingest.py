import os
import hashlib
import feedparser
import urllib.parse
import requests
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# ---- Config ----
DATA_DIR = "data"
DB_PATH = "./vector_store"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def compute_text_hash(text: str) -> str:
    """Compute hash for a single chunk of text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def compute_file_hash(file_path: str) -> str:
    """Compute hash for an entire file (PDF)."""
    if not os.path.isfile(file_path):
        raise ValueError(f"compute_file_hash expected a file, got directory: {file_path}")
    BUF_SIZE = 65536  # read in 64kb chunks
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(BUF_SIZE):
            md5.update(chunk)
    return md5.hexdigest()


# --- arXiv Search ---
def arxiv_search(query, max_results=5):
    encoded_query = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&max_results={max_results}"
    feed = feedparser.parse(url)
    results = []
    for entry in feed.entries:
        pdf_url = None
        for link in entry.links:
            if link.get("title") == "pdf":
                pdf_url = link.href
                break
        if pdf_url:
            results.append({
                "title": entry.title,
                "summary": entry.summary,
                "pdf_url": pdf_url,
                "authors": [a.name for a in entry.authors],
                "published": entry.published
            })
    return results


def fetch_pdf(entry, topic: str):
    """Download and save a PDF from arXiv entry if not already stored."""
    pdf_url = entry.get("pdf_url")
    if not pdf_url:
        return None

    response = requests.get(pdf_url)
    response.raise_for_status()

    # Compute hash from content (not file path)
    file_hash = hashlib.md5(response.content).hexdigest()
    file_path = Path(DATA_DIR) / f"{file_hash}.pdf"

    if not file_path.exists():
        with open(file_path, "wb") as f:
            f.write(response.content)

    return str(file_path)


def parse_and_chunk(file_path: str, topic: str):
    """Load a single PDF and chunk into smaller sections."""
    docs = []

    path = Path(file_path)
    if path.is_file():
        files = [path]
    else:
        files = list(path.glob("*.pdf"))

    for file in files:
        loader = PyPDFLoader(str(file))
        pdf_docs = loader.load()

        # Add topic + source metadata
        for d in pdf_docs:
            d.metadata.update({
                "source": str(file),
                "topic": topic
            })
        docs.extend(pdf_docs)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return chunks


# --- Store embeddings with deduplication ---
def store_embeddings(chunks, file_path=None, db_path=DB_PATH):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    # Collect existing hashes
    existing_docs = vectorstore.get(include=["metadatas"])
    existing_file_hashes = set()
    existing_chunk_hashes = set()

    if "metadatas" in existing_docs:
        for meta in existing_docs["metadatas"]:
            if "file_hash" in meta:
                existing_file_hashes.add(meta["file_hash"])
            if "chunk_hash" in meta:
                existing_chunk_hashes.add(meta["chunk_hash"])

    # --- File-level deduplication ---
    file_hash = compute_file_hash(file_path) if file_path else None
    if file_hash and file_hash in existing_file_hashes:
        print(f"ℹ️ Skipping {file_path} — already ingested.")
        return

    # --- Chunk-level deduplication ---
    new_chunks = []
    for chunk in chunks:
        chash = compute_text_hash(chunk.page_content)
        if chash not in existing_chunk_hashes:
            chunk.metadata["chunk_hash"] = chash
            if file_hash:
                chunk.metadata["file_hash"] = file_hash
                chunk.metadata["file_name"] = Path(file_path).name
            new_chunks.append(chunk)

    if new_chunks:
        vectorstore.add_documents(new_chunks)
        vectorstore.persist()
        print(f"✅ Added {len(new_chunks)} new chunks from {file_path}.")
    else:
        print("ℹ️ No new chunks to add — already ingested.")


def ingest_topic(topic: str, max_results: int = 5):
    """Search arXiv, fetch PDFs, parse + chunk, then embed."""
    entries = arxiv_search(topic, max_results=max_results)

    all_chunks = []
    for entry in entries:
        file_path = fetch_pdf(entry, topic)
        if file_path:
            chunks = parse_and_chunk(file_path, topic)
            all_chunks.extend(chunks)
            store_embeddings(chunks, file_path=file_path, db_path=DB_PATH)

    print(f"📚 Ingested {len(all_chunks)} chunks for topic: {topic}")
    return all_chunks


if __name__ == "__main__":
    topic = "neutron stars"
    ingest_topic(topic, max_results=3)
