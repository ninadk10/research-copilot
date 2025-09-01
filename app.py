import streamlit as st
import os
from ingest import load_and_split_pdfs, store_embeddings
from query import generate_answer
from datetime import datetime
import pandas as pd

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

st.set_page_config(page_title="Research Copilot", layout="wide")
st.title("📄 Research Copilot (LangChain)")

# ---- Document Upload Section ----
st.header("1. Upload Documents")
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    os.makedirs("data", exist_ok=True)
    for uploaded_file in uploaded_files:
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {len(uploaded_files)} files.")

    if st.button("Process Documents"):
        chunks = load_and_split_pdfs("data/")
        store_embeddings(chunks, DB_PATH)
        st.success("Documents processed and stored.")

# ---- Query Section ----
st.header("2. Ask a Question")
query = st.text_input("Enter your question:")

if st.button("Get Answer") and query:
    answer, sources = generate_answer(query)

    st.subheader("AI Response")
    st.write(answer)

    st.subheader("Sources Used")
    for doc in sources:
        st.caption(doc.metadata.get("source", "Unknown"))

    # ---- Evaluation Section ----
    st.header("3. Evaluate Answer")
    eval_mode = st.selectbox(
        "Choose Evaluation Method",
        ["Manual Feedback Only", "RAGAS", "TruLens"],
        index=0
    )

    if eval_mode == "Manual Feedback Only":
        feedback = st.radio("How useful was this answer?", ["Excellent", "Good", "Fair", "Poor"])

    elif eval_mode == "RAGAS" and ragas_available:
        st.write("Running RAGAS metrics...")
        faith_score = faithfulness.run(sources, answer)
        rel_score = answer_relevance.run(query, answer)
        st.metric("Faithfulness", f"{faith_score:.2f}")
        st.metric("Relevance", f"{rel_score:.2f}")
        feedback = f"Faithfulness: {faith_score:.2f}, Relevance: {rel_score:.2f}"

    elif eval_mode == "TruLens" and trulens_available:
        st.write("Running TruLens feedback...")
        feedback_fn = Feedback(lambda _: 1.0)  # placeholder function
        feedback_score = feedback_fn(None)
        st.metric("TruLens Feedback Score", f"{feedback_score:.2f}")
        feedback = f"TruLens Score: {feedback_score:.2f}"

    else:
        st.warning(f"{eval_mode} not available. Install required library.")
        feedback = None

    # Save feedback
    if feedback and st.button("Save Feedback"):
        os.makedirs("feedback", exist_ok=True)
        df = pd.DataFrame([{
            "timestamp": datetime.now(),
            "query": query,
            "answer": answer,
            "feedback": feedback
        }])
        feedback_file = "feedback/feedback_log.csv"
        if os.path.exists(feedback_file):
            df.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            df.to_csv(feedback_file, index=False)
        st.success("Feedback saved.")
