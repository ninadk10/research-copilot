import streamlit as st
from langchain_community.llms import Ollama
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import re
from langchain.chat_models import ChatOpenAI
import textstat


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


DB_PATH = "./vector_store"
DATA_DIR = "data"

st.set_page_config(page_title="Research Copilot Agent", layout="wide")
st.title("📄 Research Copilot (LangChain Agent)")

# ---- Document Upload Section ----
st.header("1. Upload Documents (optional)")
uploaded_files = st.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
)

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

    progress = st.progress(0)
    status = st.empty()

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

    progress.progress(20)

    # Step 1: Planner
    with st.expander("1️⃣ Planning: Sub-questions"):
        sub_questions = planner_tool(user_query)
        for i, q in enumerate(sub_questions, 1):
            st.write(f"{i}. {q}")
    progress.progress(30)

   
  # Step 2: Researcher

    # Step 2: Researcher

    findings = []
    all_sources = []

    with st.expander("2️⃣ Researcher: Findings per Sub-question"):
        for q in sub_questions:
            st.write(f"**Q:** {q}")

            answer_text, src_list = researcher_tool(q)

            # display answer
            st.write(answer_text)
            st.write(src_list)
            findings.append(answer_text)

            # display sources
            if src_list:
                st.markdown("**Sources:**")
                for src in src_list:
                    if src.startswith("http") or "arxiv" in src.lower():
                        st.markdown(f"- [{src}]({src})")
                    else:
                        st.markdown(f"- {src}")

            # accumulate
            all_sources.extend(src_list)

    compiled_findings = "\n\n".join(f for f in findings)
    progress.progress(40)



    # # Step 3: Writer
    # with st.expander("3️⃣ Writer: Draft Report"):
    #     draft_report = writer_tool(compiled_findings)
    #     st.write(draft_report)
    # progress.progress(60)

    # # Step 4: Critic
    # with st.expander("4️⃣ Critic: Final Report"):
    #     final_report = critic_tool(draft_report)
    #     st.success(final_report)
    # progress.progress(80)

    # # ---- Evaluation Section ----
    # st.header("3. Evaluate Answer")

    # from langchain_community.embeddings import OllamaEmbeddings
    # from sklearn.metrics.pairwise import cosine_similarity
    # import numpy as np
    # from collections import Counter
    # from langchain_community.llms import Ollama

    # llm = Ollama(model="llama3.2", temperature=0.2)

    # # 🧠 1️⃣ LLM Self-Evaluation (Faithfulness / Relevance / Clarity / Coverage)
    # st.subheader("🔍 LLM Self-Evaluation")

    # criteria = [
    #     "Factual accuracy",
    #     "Clarity and coherence",
    #     "Relevance to the question",
    #     "Coverage (completeness)",
    # ]

    # eval_prompt = f"""
    # You are an expert evaluator.
    # Evaluate the following research report on a scale of 1–10 for each of these criteria:
    # {", ".join(criteria)}.

    # Question: {user_query}
    # Answer: {final_report}

    # Respond in strict JSON format with no preamble or explanation. Return the JSON only:
    # {{"Factual accuracy": <score>, "Clarity and coherence": <score>, "Relevance": <score>, "Coverage": <score>}}
    # """

    # try:
    #     llm_response = llm.invoke(eval_prompt)
    #     st.json(llm_response)
    # except Exception as e:
    #     st.warning(f"❌ LLM evaluation failed: {e}")

    # # 🔢 2️⃣ Embedding-Based Semantic Similarity
    # st.subheader("🧩 Embedding-Based Semantic Similarity")

    # try:
    #     embedder = OllamaEmbeddings(model="llama3.2")
    #     context_emb = np.mean(embedder.embed_documents([compiled_findings]), axis=0)
    #     answer_emb = embedder.embed_query(final_report)
    #     similarity = cosine_similarity([context_emb], [answer_emb])[0][0]
    #     st.info(f"Semantic similarity score: {similarity:.3f}")
    # except Exception as e:
    #     st.warning(f"⚠️ Semantic similarity evaluation failed: {e}")

    # # 📖 3️⃣ Readability Metrics
    # st.subheader("📚 Readability Metrics")

    # try:
    #     readability = textstat.flesch_reading_ease(final_report)
    #     grade_level = textstat.flesch_kincaid_grade(final_report)
    #     st.info(f"Flesch Reading Ease: {readability:.2f}")
    #     st.info(f"Flesch-Kincaid Grade Level: {grade_level:.2f}")
    # except Exception as e:
    #     st.warning(f"⚠️ Readability evaluation failed: {e}")

    # # 🔤 4️⃣ Keyword Overlap
    # st.subheader("🔡 Keyword Overlap Metric")

    # def keyword_overlap(a, b):
    #     words_a = set(a.lower().split())
    #     words_b = set(b.lower().split())
    #     return (
    #         len(words_a & words_b) / len(words_a | words_b) if words_a | words_b else 0
    #     )

    # try:
    #     overlap = keyword_overlap(compiled_findings, final_report)
    #     st.info(f"Keyword overlap: {overlap:.3f}")
    # except Exception as e:
    #     st.warning(f"⚠️ Keyword overlap evaluation failed: {e}")

    # progress.progress(100)
    # st.success("Research pipeline completed!")

    # with st.expander("📚 Sources Used"):
    #     if all_sources:
    #         st.write("### Documents referenced:")

    #         # Deduplicate sources
    #         seen = set()
    #         for src in all_sources:
    #             ref = src.get("source") or src.get("file_path") or src.get("title")
    #             if ref and ref not in seen:
    #                 seen.add(ref)

    #                 # Format clickable arXiv links
    #                 if "arxiv" in ref.lower():
    #                     st.markdown(f"- [{ref}]({ref})")
    #                 else:
    #                     st.markdown(f"- {ref}")
    #     else:
    #         st.write("No sources collected.")
