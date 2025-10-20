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
    description="Decompose a research question into exactly 3 clear, concise sub-questions."
)


# -----------------------
# Researcher Tool
# -----------------------
# Replace the old researcher_tool with this function in agent_pipeline.py

def researcher_tool(sub_question: str) -> str:
    """
    Run retrieval + QA for one sub-question and return a formatted string.
    Robust to different return shapes from generate_answer (tuple/dict/string)
    and to source docs being dicts or Document objects.
    """
    try:
        # call the QA/retrieval function (imported as generate_answer)
        result = generate_answer(sub_question)

        # Normalize result into (answer, sources)
        answer = None
        sources = []

        if result is None:
            answer = None
            sources = []
        elif isinstance(result, tuple) or isinstance(result, list):
            # common case: (answer, source_docs)
            if len(result) >= 2:
                answer, sources = result[0], result[1]
            else:
                answer = result[0] if result else None
        elif isinstance(result, dict):
            # some chains return dicts with keys like 'result' and 'source_documents'
            answer = result.get("result") or result.get("answer") or result.get("output") or None
            sources = result.get("source_documents") or result.get("source_docs") or result.get("source_documents", []) or []
        else:
            # string (rare)
            answer = str(result)
            sources = []

        # If answer is empty/None, try to build a fallback from retrieved passages
        if not answer or (isinstance(answer, str) and not answer.strip()):
            snippets = []
            for doc in (sources or []):
                if isinstance(doc, dict):
                    txt = doc.get("page_content") or doc.get("text") or ""
                else:
                    txt = getattr(doc, "page_content", None) or getattr(doc, "page", None) or ""
                if txt:
                    snippets.append(txt.strip())
            if snippets:
                # Use top 3 snippets as a fallback summary
                answer = "(No direct LLM answer returned. Fallback assembled from retrieved passages.)\n\n" + "\n\n---\n\n".join(snippets[:3])
            else:
                answer = "No answer could be generated and no retrieved passages available."

        # Build a compact sources list (unique)
        src_names = []
        for doc in (sources or []):
            if isinstance(doc, dict):
                meta = doc.get("metadata") or {}
            else:
                meta = getattr(doc, "metadata", {}) or {}
            src = meta.get("source") or meta.get("file_name") or meta.get("url") or None
            if src:
                src_names.append(str(src))
        # unique-preserve-order
        seen = set()
        unique_srcs = [x for x in src_names if not (x in seen or seen.add(x))]

        # Build formatted output
        formatted = answer.strip() if isinstance(answer, str) else str(answer)
        if unique_srcs:
            formatted += "\n\nSources:\n" + "\n".join(unique_srcs)

        return formatted

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        # return readable error to UI (avoids returning None)
        return f"Error in researcher_tool: {e}\n{tb}"


researcher = Tool(
    name="Researcher",
    func=researcher_tool,
    description="Use this tool to research sub-questions. It will fetch papers, embed them, and answer queries."
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
    description="Synthesize research results into a coherent report."
)


# -----------------------
# Critic Tool
# -----------------------
def critic_tool(draft_report: str) -> str:
    prompt = f"Critically review this research report and improve clarity, coverage, and correctness:\n{draft_report}"
    return llm_call(prompt)


critic = Tool(
    name="Critic",
    func=critic_tool,
    description="Review and improve research reports."
)


# -----------------------
# Agent (LangChain-style)
# -----------------------
tools = [planner, researcher, writer, critic]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
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
