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



def load_pdfs(folder):
    all_texts = []
    for file in folder:
        if file     
        reader = PdfReader(file)
        number_of_pages = len(reader.pages)
        page = reader.pages[0]
        text = page.extract_text()
        chunk_text(text)
    return all_texts

def chunk_text(text, size=500):
    chunked_text = []
    for word in text:

        chunked_text.append(word)





def embed_chunks(chunks):
    # sentence transformers for embeddings

def store_embeddings(chunks,embeddings):
    # chroma db for vector store