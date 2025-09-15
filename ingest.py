# ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import feedparser
import urllib.parse
import requests
from pathlib import Path

DATA_DIR = "data"
DB_PATH = "./vector_store"

# search arXiv for relevant papers
def arxiv_search(query, max_results=5):
    url = f"http://export.arxiv.org/api/query?search_query={query}&max_results={max_results}"
    feed = feedparser.parse(url)
    results = []
    for entry in feed.entries:
        pdf_url = None
        #  Iterate through the links to find pdf link with title "pdf"
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
            print(f"PDF link not found for entry: {entry.title}")
    return results


# fetch and downlaod pdfs locally
def fetch_pdf(url, out_dir=DATA_DIR):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = out_path / (url.split("/")[-1] + ".pdf")
    
    r = requests.get(url)
    r.raise_for_status()
    filename.write_bytes(r.content)
    return str(filename)

# parse and chunk text
def parse_and_chunk(filename):
    """Load all PDFs from directory and split into chunks."""
    documents = []
    for filename in os.listdir(DATA_DIR):
        pdf_path = os.path.join(DATA_DIR, filename)
        if pdf_path.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, filename))
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return text_splitter.split_documents(documents)

def store_embeddings(chunks, db_path=DB_PATH):
    """Embed chunks and store in ChromaDB."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_path)
    print(f"Stored {len(chunks)} chunks in vector store at {db_path}")

# main ingest function
def ingest_topic(query, max_results=5):
    clean_query = urllib.parse.quote_plus(query)
    results = arxiv_search(clean_query, max_results=max_results)
    for paper in results:
        pdf_path = fetch_pdf(paper["pdf_url"])
        print(pdf_path)
        chunks = parse_and_chunk(pdf_path)
        store_embeddings(chunks)
    print(f"Ingested {len(results)} papers for topic: {query}")


