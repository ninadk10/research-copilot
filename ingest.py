""" 
ingest.py – Load PDFs, Split Text, Store Embeddings

Uses PyPDF for parsing, Sentence-Transformers for embeddings, ChromaDB (or FAISS) for vector storage.

Key functions:

load_pdfs(folder)

chunk_text(text, size=500)

embed_chunks(chunks)

store_embeddings(chunks, embeddings)
"""

from PyPDF2 import PdfReader
import os
from sentence_transformers import SentenceTransformer
import chromadb


client = chromadb.PersistentClient()
folder_path = os.fsencode("data/")

def load_pdfs(folder_path):
    all_texts = []
    for file in os.listdir(folder_path):
        filename = os.fsdecode(file)
        if filename.endswith(".pdf"):     
            reader = PdfReader(file)
            number_of_pages = len(reader.pages)
            for i in number_of_pages:    
                page = reader.pages[i]
                text = page.extract_text()
                all_texts.append(text)
    return all_texts


def chunk_text(text, chunk_size=500, overlap = 50):
    text_len = len(text)
    chunks = []
    start = 0
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def embed_chunks(chunks, model_name):
    # sentence transformers for embeddings
    model = SentenceTransformer(model_name) 
    embeddings = []
    for chunk in chunks:
        embedding = model.encode(chunk)
        embeddings.append(embedding)
    return embeddings

def store_embeddings(chunks,embeddings, db_path="./vector_store", collection_name="research_docs"):
    # chroma db for vector store
    collection = client.get_or_create_collection(name=collection_name)
    ids = [f"doc_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents = chunks,
        embeddings = embeddings
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB at {db_path}")


# ---- main code ------
texts = load_pdfs(folder_path)

all_chunks = []
for text in texts:
    chunks = chunk_text(text)
    all_chunks.append(chunks)

embeddings = embed_chunks(all_chunks, model_name="all-MiniLM-L6-v2")
store_embeddings(all_chunks,embeddings,)