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


class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("no message found in input")

        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(ToolMessage(content=json.dumps(tool_result), name=tool_call["name"], tool_call_id=tool_call["id"]))
        return {"messages": outputs}


def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"

    return END


tool_node = BasicToolNode(tools=[tool])

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


def stream_graph_updates(user_input: str) -> None:
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User: ", user_input)
        stream_graph_updates(user_input)
        break
