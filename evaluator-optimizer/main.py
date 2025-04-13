import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))

from pprint import pprint

from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph

from config import Settings

config = Settings()
llm = ChatOpenAI(model=config.MODEL_NAME, api_key=config.OPENAI_API_KEY)


class State(TypedDict):
    joke: str
    topic: str
    feedback: str
    funny_or_not: str


class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(description="Decide if the joke is funny or not.")
    feedback: str = Field(description="If the joke is not funny, provide feedback on how to improve it")


evaluator = llm.with_structured_output(Feedback)


def llm_call_generator(state: State) -> dict:
    """LLM generates a joke"""

    if state.get("feedback"):
        msg = llm.invoke(f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}")
    else:
        msg = llm.invoke(f"Write a joke about {state['topic']}")

    return {"joke": msg.content}


def llm_call_evaluator(state: State):
    """LLM evaluates the joke"""
    grade = evaluator.invoke(f"Grade the joke {state['joke']}")
    return {"funny_or_not": grade.grade, "feedback": grade.feedback}


def route_joke(state: State) -> Literal["Accepted", "Rejected + Feedback"]:
    if state["funny_or_not"] == "funny":
        return "Accepted"

    if state["funny_or_not"] == "not funny":
        return "Rejected + Feedback"


optimizer_builder = StateGraph(State)

optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_call_evaluator", llm_call_evaluator)

optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_call_evaluator")
optimizer_builder.add_conditional_edges("llm_call_evaluator", route_joke, {"Accepted": END, "Rejected + Feedback": "llm_call_generator"})

optimizer_workflow = optimizer_builder.compile()
state = optimizer_workflow.invoke({"topic": "Cats"})
pprint(state["joke"])
