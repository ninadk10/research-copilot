# ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sources import arxiv_search
from fetch import fetch_pdf

DATA_PATH = "data"
DB_PATH = "./vector_store"

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


def ingest_topic(query, max_results=5):
    results = arxiv_search(query, max_results=max_results)
    for paper in results:
        pdf_path = fetch_pdf(paper["pdf_url"])
        chunks = parse_and_chunk(pdf_path)
        store_embeddings(chunks)
    print(f"Ingested {len(results)} papers for topic: {query}")
