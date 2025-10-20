import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import re


from ingest import (
    arxiv_search,
    fetch_pdf,
    parse_and_chunk,
    store_embeddings,
    ingest_topic,
    compute_file_hash,
)
from agent_pipeline import (
    run_research_agent,
    planner_tool,
    researcher_tool,
    critic_tool,
    llm_call,
    writer_tool,
)

# ---- Evaluation Libraries ----
try:
    from ragas.metrics import faithfulness, answer_relevance
    ragas_available = True
except ImportError:
    ragas_available = False

try:
    from trulens_eval import Feedback
    trulens_available = True
except ImportError:
    trulens_available = False

DB_PATH = "./vector_store"
DATA_DIR = "data"

st.set_page_config(page_title="Research Copilot Agent", layout="wide")
st.title("📄 Research Copilot (LangChain Agent)")

# ---- Document Upload Section ----
st.header("1. Upload Documents (optional)")
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    os.makedirs(DATA_DIR, exist_ok=True)

    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = Path(DATA_DIR) / uploaded_file.name

        # Save uploaded PDF
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        saved_files.append(str(file_path))

    st.success(f"Uploaded {len(saved_files)} files.")

    if st.button("Process Documents"):
        for file_path in saved_files:
            file_hash = compute_file_hash(file_path)

            # Check if already in vector store (file-level dedup)
            chunks = parse_and_chunk(file_path)
            store_embeddings(chunks, file_path=file_path, db_path=DB_PATH)

        st.success("Documents processed and stored with deduplication.")

def safe_filename(name: str) -> str:
    """Sanitize string for safe filesystem usage."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

# ---- Query Section ----
st.header("2. Ask a Question")
user_query = st.text_input("Enter your research question:")

if st.button("Run Research Agent") and user_query.strip():
    st.info("Running the research pipeline...")

    # 🔍 Step 0: Ingest topic automatically (if not already in vector store)
    safe_topic = safe_filename(user_query.replace(" ", "_"))
    topic_hash_path = Path(DB_PATH) / f"{safe_topic}.lock"

    if not topic_hash_path.exists():
        st.write(f"🔄 Ingesting research papers for topic: **{user_query}** ...")
        try:
            ingest_topic(user_query, max_results=3)
            topic_hash_path.touch()  # Mark topic as ingested
            st.success("✅ Topic ingestion completed.")
        except Exception as e:
            st.warning(f"⚠️ Ingestion skipped due to error: {e}")
    else:
        st.info("✅ This topic has already been ingested — skipping re-ingestion.")

    # Step 1: Planner
    with st.expander("1️⃣ Planning: Sub-questions"):
        sub_questions = planner_tool(user_query)
        for i, q in enumerate(sub_questions, 1):
            st.write(f"{i}. {q}")

    # Step 2: Researcher
    findings = []
    with st.expander("2️⃣ Researcher: Findings per Sub-question"):
        for q in sub_questions:
            st.write(f"**Q:** {q}")
            answer = researcher_tool(q)
            st.write(answer)
            findings.append(answer)

    compiled_findings = "\n\n".join([str(f) for f in findings if f])

    # Step 3: Writer
    with st.expander("3️⃣ Writer: Draft Report"):
        draft_report = writer_tool(compiled_findings)
        st.write(draft_report)

    # Step 4: Critic
    with st.expander("4️⃣ Critic: Final Report"):
        final_report = critic_tool(draft_report)
        st.success(final_report)

    # ---- Evaluation Section ----
    st.header("3. Evaluate Answer")
    eval_mode = st.selectbox(
        "Choose Evaluation Method",
        ["Manual Feedback Only", "RAGAS", "TruLens"],
        index=0,
    )

    if eval_mode == "Manual Feedback Only":
        feedback = st.radio("How useful was this answer?", ["Excellent", "Good", "Fair", "Poor"])

    elif eval_mode == "RAGAS" and ragas_available:
        st.write("Running RAGAS metrics...")
        # In a full setup, context can be retrieved or passed separately
        faith_score = faithfulness.run([], final_report)  # placeholder empty context
        rel_sc_
