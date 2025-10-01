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
def researcher_tool(sub_question: str) -> str:
    """
    For each sub-question:
      1. Ingest relevant papers (fetch + embed).
      2. Query the vector store for an answer.
    """
    # Step 1: Fetch and store relevant docs for this sub-question
    ingest_topic(sub_question)

    # Step 2: Query the vector store for answers
    answer, sources = generate_answer(sub_question)

    # Format sources into short summary
    source_text = "\n".join([doc.metadata.get("source", "Unknown") for doc in sources])
    return f"{answer}\n\nSources:\n{source_text}"

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
