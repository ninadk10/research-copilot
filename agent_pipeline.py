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


