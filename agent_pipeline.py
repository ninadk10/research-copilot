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
    """Break down research question into sub-questions."""
    prompt = f"Break down this research question into 3-5 precise sub-questions:\n{user_query}"
    response = llm_call(prompt)

    # Parse into list
    sub_questions = [
        line.strip(" -1234567890. ")
        for line in response.split("\n")
        if line.strip()
    ]
    return sub_questions


planner = Tool(
    name="Planner",
    func=planner_tool,
    description="Decompose a research question into actionable sub-questions."
)


# -----------------------
# Researcher Tool
# -----------------------
def researcher_tool(sub_question: str) -> str:
    """
    Ingest relevant papers + query the vector store for each sub-question.
    """
    ingest_topic(sub_question)
    answer, sources = generate_answer(sub_question)
    source_text = "\n".join([doc.metadata.get("source", "Unknown") for doc in sources])
    return f"Q: {sub_question}\nA: {answer}\nSources:\n{source_text}\n"


researcher = Tool(
    name="Researcher",
    func=researcher_tool,
    description="Research sub-questions and provide answers with sources."
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
