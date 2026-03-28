"""
으뜸 딥러닝 — 14장 08절
간단한 ReAct 에이전트 (LangChain)
"""

from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import DuckDuckGoSearchRun

tools = [DuckDuckGoSearchRun()]
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "What is the weather in Seoul today?"})
