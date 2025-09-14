# agent_chain.py
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import Ollama
from langchain.tools import Tool
from query import generate_answer
from langchain.tools import Tool
from llm_utils import llm_call
from langchain.tools import Tool
from query import generate_answer

# 🔹 Choose your local model (llama3, mistral, gemma)
llm = Ollama(model="llama3.2", temperature=0.2)


def run_research_agent(user_query: str) -> str:
    """
    Input: High-level research question
    Output: Final research report
    """
    final_report = agent.run(user_query)
    return final_report

# llm_utils.py

# Create one shared instance of the local LLM
llm = Ollama(model="llama3.2", temperature=0.2)

def llm_call(prompt: str) -> str:
    """Utility function to query the local Ollama model."""
    return llm.invoke(prompt)

# Researcher Tool: retrieves and summarizes documents
def researcher_tool(query: str) -> str:
    answer, sources = generate_answer(query)
    # Optionally format sources into short summary
    source_text = "\n".join([doc.metadata.get("source", "Unknown") for doc in sources])
    return f"{answer}\n\nSources:\n{source_text}"

researcher = Tool(
    name="Researcher",
    func=researcher_tool,
    description="Use this tool to research sub-questions and retrieve evidence from stored papers."
)


def planner_tool(user_query: str) -> str:
    prompt = f"Break down this research question into 3-5 precise sub-questions:\n{user_query}"
    return llm_call(prompt)

planner = Tool(
    name="Planner",
    func=planner_tool,
    description="Decompose a high-level research question into actionable sub-questions."
)


def writer_tool(research_results: str) -> str:
    prompt = f"Write a structured research report from the following findings:\n{research_results}"
    return llm_call(prompt)

writer = Tool(
    name="Writer",
    func=writer_tool,
    description="Synthesizes research results into a coherent report with sections."
)


def critic_tool(draft_report: str) -> str:
    prompt = f"Critically review this research report and improve clarity, coverage, and correctness:\n{draft_report}"
    return llm_call(prompt)

critic = Tool(
    name="Critic",
    func=critic_tool,
    description="Reviews a draft research report and suggests improvements."
)


tools = [planner, researcher, writer, critic]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)