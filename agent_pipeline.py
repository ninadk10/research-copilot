# agent_chain.py
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

from query import generate_answer
from ingest import ingest_topic


# -----------------------
# Shared LLM
# -----------------------
llm = Ollama(model="llama3.2", temperature=0.2)


def llm_call(prompt: str) -> str:
    """Utility function to query the local Ollama model."""
    return llm.invoke(prompt)


# -----------------------
# Planner Tool
# -----------------------
def planner_tool(user_query: str) -> list[str]:
    """Break down a research question into 3 concise sub-questions."""
    prompt = f"""
You are a research planning assistant.

Given the research question below, generate exactly 3 concise and relevant sub-questions.
Each sub-question should help break down the main question into manageable parts.

Format the output cleanly as:
1. <sub-question 1>
2. <sub-question 2>
3. <sub-question 3>

Avoid any introductions, explanations, or extra commentary.
Return only the numbered sub-questions.

Research Question: {user_query}
"""

    response = llm_call(prompt)

    # Parse the numbered list into clean sub-questions
    sub_questions = [
        line.strip(" -1234567890. ")
        for line in response.split("\n")
        if line.strip() and any(c.isalpha() for c in line)
    ]

    # Ensure only top 3 sub-questions are returned
    return sub_questions[:3]


planner = Tool(
    name="Planner",
    func=planner_tool,
    description="Decompose a research question into exactly 3 clear, concise sub-questions.",
)


# -----------------------
# Researcher Tool
# -----------------------
# Replace the old researcher_tool with this function in agent_pipeline.py

def researcher_tool(sub_question: str):
    """
    Retrieval + QA for one sub-question.
    Returns (answer_text, unique_source_list)
    Always returns a source list, even if empty.
    """

    try:
        raw = generate_answer(sub_question)

        # ------------------------------------
        # Normalize results into (answer, docs)
        # ------------------------------------
        answer = None
        source_docs = []

        # Case 1: tuple or list -> (answer, docs)
        if isinstance(raw, (tuple, list)):
            if len(raw) >= 2:
                answer, source_docs = raw[0], raw[1]
            else:
                answer = raw[0]

        # Case 2: dict -> typical LangChain format
        elif isinstance(raw, dict):
            answer = raw.get("result") or raw.get("answer") or raw.get("output")
            source_docs = raw.get("source_documents") or raw.get("source_docs") or []

        # Case 3: plain string
        else:
            answer = str(raw)

        # ------------------------------------
        # Extract answer fallback
        # ------------------------------------
        if not answer:
            answer = "No generated answer."

        # ------------------------------------
        # Extract metadata from source documents
        # ------------------------------------
        extracted_sources = []

        for doc in source_docs:
            meta = {}

            if isinstance(doc, dict):
                meta = doc.get("metadata", {})
            else:
                meta = getattr(doc, "metadata", {}) or {}

            # MOST COMMON METADATA KEYS ACROSS LANGCHAIN + CHROMA
            possible_keys = [
                "source",
                "file_path",
                "file_name",
                "title",
                "url",
                "link",
                "pdf_file",
                "arxiv_id",
                "document_id",
            ]

            src_val = None
            for key in possible_keys:
                if key in meta and meta[key]:
                    src_val = str(meta[key])
                    break

            # Last resort: look for any URL-ish value
            if not src_val:
                for k, v in meta.items():
                    if isinstance(v, str) and ("http" in v or "arxiv" in v.lower()):
                        src_val = v
                        break

            if src_val:
                extracted_sources.append(src_val)

        # Deduplicate
        dedup_sources = list(dict.fromkeys(extracted_sources))

        # ------------------------------------
        # Final output
        # ------------------------------------
        formatted = answer.strip() if isinstance(answer, str) else str(answer)

        return formatted, dedup_sources

    except Exception as e:
        import traceback
        return f"Error in researcher_tool: {e}\n\n{traceback.format_exc()}", []



researcher = Tool(
    name="Researcher",
    func=researcher_tool,
    description="Use this tool to research sub-questions. It will fetch papers, embed them, and answer queries.",
)


# -----------------------
# Writer Tool
# -----------------------
def writer_tool(research_results: str) -> str:
    prompt = f"Write a structured research report from the following findings:\n{research_results}"
    return llm_call(prompt)


writer = Tool(
    name="Writer",
    func=writer_tool,
    description="Synthesize research results into a coherent report.",
)


# -----------------------
# Critic Tool
# -----------------------
def critic_tool(draft_report: str) -> str:
    prompt = f"Critically review this research report and improve clarity, coverage, and correctness:\n{draft_report}"
    return llm_call(prompt)


critic = Tool(
    name="Critic", func=critic_tool, description="Review and improve research reports."
)


# -----------------------
# Agent (LangChain-style)
# -----------------------
tools = [planner, researcher, writer, critic]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


# -----------------------
# Orchestrated Pipeline
# -----------------------
def run_research_agent(user_query: str) -> str:
    """
    End-to-end pipeline:
      1. Plan → sub-questions
      2. Research → findings
      3. Write → draft
      4. Critic → final report
    """
    sub_questions = planner_tool(user_query)

    findings = []
    for q in sub_questions:
        findings.append(researcher_tool(q))
    compiled_findings = "\n\n".join(findings)

    draft_report = writer_tool(compiled_findings)
    final_report = critic_tool(draft_report)

    return final_report
