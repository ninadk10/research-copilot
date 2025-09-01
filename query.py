# query.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub  # or OpenAI() if using GPT-4
import os

DB_PATH = "./vector_store"

def create_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return vectordb.as_retriever()

def generate_answer(query):
    retriever = create_retriever()

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.2, "max_length": 512}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain(query)
    return result["result"], result["source_documents"]
