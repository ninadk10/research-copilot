# query.py
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub  # or OpenAI() if using GPT-4
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

DB_PATH = "./vector_store"

def create_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return vectordb.as_retriever()

def generate_answer(query):
    retriever = create_retriever()

    # Access token automatically from environment
    llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",   # 👈 force the task
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain(query)
    return result["result"], result["source_documents"]


x,y = generate_answer("LLM")
print(x,y)