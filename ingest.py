# ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import feedparser
import requests
from pathlib import Path

DATA_PATH = "data"
DB_PATH = "./vector_store"

# search arXiv for relevant papers
def arxiv_search(query, max_results=5):
    url = f"http://export.arxiv.org/api/query?search_query={query}&max_results={max_results}"
    feed = feedparser.parse(url)
    results = []
    for entry in feed.entries:
        results.append({
            "title": entry.title,
            "summary": entry.summary,
            "pdf_url": entry.links[1].href,  # link[0] is abstract, link[1] is PDF
            "authors": [a.name for a in entry.authors],
            "published": entry.published
        })
    return results


# fetch and downlaod pdfs locally
def fetch_pdf(url, out_dir="data/papers"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = out_path / (url.split("/")[-1] + ".pdf")
    
    r = requests.get(url)
    r.raise_for_status()
    filename.write_bytes(r.content)
    return str(filename)

# parse and chunk text
def load_and_split_pdfs(pdf_dir):
    """Load all PDFs from directory and split into chunks."""
    documents = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, file))
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
    results = arxiv_search(query, max_results=max_results)
    for paper in results:
        pdf_path = fetch_pdf(paper["pdf_url"])
        chunks = parse_and_chunk(pdf_path)
        store_embeddings(chunks)
    print(f"Ingested {len(results)} papers for topic: {query}")




