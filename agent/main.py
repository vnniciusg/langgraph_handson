import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import START, END, StateGraph

from config import Settings

config = Settings()
llm = ChatOpenAI(name=config.MODEL_NAME, api_key=config.OPENAI_API_KEY)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b


tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


def llm_call(state: MessagesState) -> dict:
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")] + state["messages"]
            )
        ]
    }


def tool_node(state: dict) -> dict:
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_contiue(state: MessagesState) -> Literal["environment", END]:
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "Action"

    return END


agent_builder = StateGraph(MessagesState)

agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_contiue, {"Action": "environment", END: END})
agent_builder.add_edge("environment", "llm_call")

agent = agent_builder.compile()

messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for message in messages["messages"]:
    message.pretty_print()
