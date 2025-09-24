🧑‍🔬 Agentic AI Research Copilot

An open-source AI-powered research assistant that ingests academic papers, reasons with an agentic workflow, and produces structured literature reviews.

Built with LangChain, ChromaDB, Streamlit, and evaluated using RAGAS + TruLens.
🚀 Features

📥 Ingestion Pipeline – Fetch papers from arXiv, parse, chunk, embed, and store in ChromaDB.

🔎 Retriever + Agent Workflow – Planner → Researcher → Writer → Critic loop, mimicking human research.

🖥️ Streamlit UI – Ask questions, get structured answers, and explore supporting passages.

📊 Evaluation Layer – Hybrid RAGAS + TruLens evaluation for faithfulness and groundedness.

⚡ Extensible – Add more sources (PubMed, Semantic Scholar), swap embedding models, or extend agents.

🏗️ Project Structure
research-copilot/
│
├── ingest.py        # Ingests and embeds research papers
├── query.py         # Queries the vector store
├── app.py           # Streamlit UI entry point
├── main.py          # Orchestrates the agentic workflow
│
├── copilot/
│   ├── agents/      # Planner, Researcher, Writer, Critic chains
│   ├── tools/       # Retriever + helper functions
│   ├── eval/        # RAGAS + TruLens evaluation
│   └── utils/       # Shared helpers
│
├── requirements.txt / pyproject.toml
└── README.md

🔧 Installation

Clone the repo:

git clone https://github.com/yourusername/research-copilot.git
cd research-copilot


Create a virtual environment:

uv venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)


Install dependencies:

uv pip install -r requirements.txt


(Optional) Install PyTorch CPU version:

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

▶️ Usage
Ingest papers
python ingest.py --query "large language models in healthcare" --max_results 50

Query the knowledge base
python query.py --question "What are recent applications of LLMs in healthcare?"

Run full copilot (Streamlit UI)
streamlit run app.py

🧩 Evaluation

Toggle between RAGAS and TruLens in app.py to validate responses.

RAGAS → faithfulness, relevance, context precision.

TruLens → groundedness, correctness, transparency.

📚 Articles

This project is documented in a 3-part blog series:

System Design & Data Ingestion

Agentic Workflow: Planner → Researcher → Writer → Critic

Evaluation & Streamlit UI

🛠️ Tech Stack

LangChain – Orchestration framework

ChromaDB – Vector database

SentenceTransformers – Embeddings

Streamlit – Web UI

RAGAS / TruLens – Evaluation

🌱 Roadmap

 Add PubMed & Semantic Scholar ingestion

 Support multimodal inputs (tables, figures)

 Improve caching & retrieval filters

 Collaboration features (shareable reports)

🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR.

📄 License

MIT License © 2025 [Your Name]

✨ This project is a work-in-progress experiment in agentic AI for research. Feedback and ideas are very welcome!