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



folder_path = os.fsencode("")

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





def embed_chunks(chunks):
    # sentence transformers for embeddings

def store_embeddings(chunks,embeddings):
    # chroma db for vector store