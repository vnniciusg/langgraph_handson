import sys
import json
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from config import Settings


@dataclass(kw_only=True)
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


graph_builder = StateGraph(State)

config = Settings()
llm = ChatOpenAI(model=config.MODEL_NAME, api_key=config.OPENAI_API_KEY)

tool = TavilySearchResults(max_results=2, tavily_api_key=config.TAVILY_API_KEY)
tools = [tool]

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


tool_node = ToolNode(tools=[tool])

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


inputs = ["Hi there! My name is Will.", "Remember my name?"]
for user_input in inputs:
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        {'configurable': {'thread_id': "1"}},
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()
