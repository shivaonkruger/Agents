import os
from typing import List, TypedDict, Union

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file or environment.")


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


def process(state: AgentState) -> AgentState:
    """Run the Groq-backed agent node."""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    print("CURRENT STATE:", state["messages"])

    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")