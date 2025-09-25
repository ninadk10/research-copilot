import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM


DB_PATH = "./vector_store"

# Initialize embeddings + vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def create_retriever():
    """Load persisted ChromaDB and return a retriever."""
    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    return vectordb.as_retriever(search_kwargs={"k": 4})


def generate_answer(query: str):
    """Query vector DB and generate an answer with sources + topics."""
    retriever = create_retriever()

    # Local Ollama model
    llm = OllamaLLM(model="llama3.2", temperature=0.2)

    # Custom prompt template
    prompt_template = """
    You are a helpful research assistant.
    Use the provided context to answer the question.
    If context is insufficient, say you are not sure.

    Question: {question}

    Context:
    {context}

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    result = qa_chain({"query": query})

    answer = result["result"]
    source_docs = result["source_documents"]

    # Extract metadata
    sources = []
    for doc in source_docs:
        metadata = doc.metadata
        sources.append({
            "source": metadata.get("source", "Unknown"),
            "topic": metadata.get("topic", "Unknown"),
            "snippet": doc.page_content[:200] + "..."  # preview
        })

    return answer, sources


if __name__ == "__main__":
    q = "What are neutron stars?"
    answer, sources = generate_answer(q)
    print("Answer:\n", answer)
    print("\nSources:")
    for s in sources:
        print(f"- {s['topic']} | {s['source']}\n  {s['snippet']}")
