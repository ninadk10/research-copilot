# agent_chain.py
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI  # or HuggingFaceHub, Ollama, etc.
from tools import planner, researcher, writer, critic

llm = ChatOpenAI(temperature=0.2, model_name="gpt-4")  # choose your LLM

tools = [planner, researcher, writer, critic]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


def run_research_agent(user_query: str) -> str:
    """
    Input: High-level research question
    Output: Final research report
    """
    final_report = agent.run(user_query)
    return final_report


# tools.py
from langchain.tools import Tool
from query import generate_answer

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



# planner_tool.py
from langchain.tools import Tool

def planner_tool(user_query: str) -> str:
    # LLM call to decompose question into sub-questions
    prompt = f"Break down this research question into 3-5 precise sub-questions:\n{user_query}"
    sub_questions = llm_call(prompt)  # wrap your LLM call here
    return sub_questions

planner = Tool(
    name="Planner",
    func=planner_tool,
    description="Decompose a high-level research question into actionable sub-questions."
)


# writer_tool.py
from langchain.tools import Tool

def writer_tool(research_results: str) -> str:
    prompt = f"Write a structured research report from the following findings:\n{research_results}"
    report = llm_call(prompt)
    return report

writer = Tool(
    name="Writer",
    func=writer_tool,
    description="Synthesizes research results into a coherent report with sections."
)


# critic_tool.py
from langchain.tools import Tool

def critic_tool(draft_report: str) -> str:
    prompt = f"Critically review this research report and improve clarity, coverage, and correctness:\n{draft_report}"
    improved_report = llm_call(prompt)
    return improved_report

critic = Tool(
    name="Critic",
    func=critic_tool,
    description="Reviews a draft research report and suggests improvements."
)
