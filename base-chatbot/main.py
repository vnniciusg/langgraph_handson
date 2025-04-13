import sys
import json
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.types import Command, interrupt

from config import Settings


@dataclass(kw_only=True)
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


graph_builder = StateGraph(State)


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human"""
    human_response = interrupt({"query": query})
    return human_response["data"]


config = Settings()
llm = ChatOpenAI(model=config.MODEL_NAME, api_key=config.OPENAI_API_KEY)

tool = TavilySearchResults(max_results=2, tavily_api_key=config.TAVILY_API_KEY)
tools = [tool, human_assistance]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> dict:
    message = llm_with_tools.invoke(state["messages"])
    assert (len(message.tool_calls) <= 1)
    return {"messages": [message]}


tool_node = ToolNode(tools=tools)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()