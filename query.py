# query.py
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

DB_PATH = "./vector_store"

def create_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return vectordb.as_retriever()

def generate_answer(query):
    retriever = create_retriever()

    # 🔹 Use Ollama LLM locally
    llm = Ollama(model="llama3.2")  # or "mistral", "gemma", etc.

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})
    return result["result"], result["source_documents"]

if __name__ == "__main__":
    answer, sources = generate_answer("What is an LLM?")
    print("Answer:", answer)
    print("Sources:", sources)
