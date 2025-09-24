import os
import hashlib
import feedparser
import urllib.parse
import requests
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data"
DB_PATH = "./vector_store"

# --- Hash helpers ---
def compute_text_hash(text: str) -> str:
    """Compute hash for a single chunk of text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def compute_file_hash(file_path: str) -> str:
    """Compute hash for an entire file (PDF)."""
    BUF_SIZE = 65536  # read in 64kb chunks
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(BUF_SIZE):
            md5.update(chunk)
    return md5.hexdigest()


# --- Search arXiv for relevant papers ---
def arxiv_search(query, max_results=5):
    url = f"http://export.arxiv.org/api/query?search_query={query}&max_results={max_results}"
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
        else:
            print(f"⚠️ PDF link not found for entry: {entry.title}")
    return results


# --- Fetch and download PDFs locally ---
def fetch_pdf(url, out_dir=DATA_DIR):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = out_path / (url.split("/")[-1] + ".pdf")
    if filename.exists():
        print(f"ℹ️ File already downloaded: {filename}")
        return str(filename)

    r = requests.get(url)
    r.raise_for_status()
    filename.write_bytes(r.content)
    return str(filename)


# --- Parse and chunk text from PDFs ---
def parse_and_chunk(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return text_splitter.split_documents(documents)


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


# --- Main ingest function ---
def ingest_topic(query, max_results=5):
    clean_query = urllib.parse.quote_plus(query)
    results = arxiv_search(clean_query, max_results=max_results)

    for paper in results:
        pdf_path = fetch_pdf(paper["pdf_url"])
        chunks = parse_and_chunk(pdf_path)
        store_embeddings(chunks, file_path=pdf_path)

    print(f"📚 Ingested {len(results)} papers for topic: {query}")
