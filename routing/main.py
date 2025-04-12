import sys
from pathlib import Path
from pprint import pprint

sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage

from config import Settings

config = Settings()
llm = ChatOpenAI(model=config.MODEL_NAME, api_key=config.OPENAI_API_KEY)


class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(None, description="The next step in the routing process")


router = llm.with_structured_output(Route)


class State(TypedDict):
    input: str
    decision: str
    output: str


def llm_call_1(state: State) -> dict:
    """Write a story"""
    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_2(state: State) -> dict:
    """Write a joke"""
    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_3(state: State) -> dict:
    """Write a poem"""
    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_router(state: State) -> dict:
    """Route the input the appropriate node"""

    decision = router.invoke(
        [SystemMessage(content="Route the input story, joke, or poem based on the user's request."), HumanMessage(content=state["input"])]
    )

    return {"decision": decision.step}


def route_decision(state: State) -> Literal["llm_call_1", "llm_call_2", "llm_call_3"]:
    if state["decision"] == "story":
        return "llm_call_1"
    if state["decision"] == "joke":
        return "llm_call_2"
    if state["decision"] == "poem":
        return "llm_call_3"


router_builder = StateGraph(State)

router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)

router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

router_workflow = router_builder.compile()

state = router_workflow.invoke({"input": "Write me a joke about cats"})
pprint(state["output"])
