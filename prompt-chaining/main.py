import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Literal

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

from config import Settings


class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str


settings = Settings()
llm = ChatOpenAI(model=settings.MODEL_NAME, api_key=settings.OPENAI_API_KEY)


def generate_joke(state: State) -> dict:
    """First LLM call to generate initial joke"""

    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}


def check_punchline(state: State) -> Literal["Fail", "Pass"]:
    """Gate function to check if the joke has a punchline"""
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Fail"

    return "Pass"


def improve_joke(state: State) -> dict:
    """Second LLM call to improve the joke"""
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}


def polish_joke(state: State) -> dict:
    """Third LLM call for final polish"""
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}


workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

workflow.add_edge(START, "generate_joke")
workflow.add_conditional_edges("generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END})
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

chain = workflow.compile()

display(Image(chain.get_graph().draw_mermaid_png()))
    
state = chain.invoke({"topic": "cats"})
print(f"Initial joke: {state['joke']}")
print("\n--- --- ---\n")
if "improved_joke" in state:
    print(f"Improved joke: {state['improved_joke']}")
    print("\n--- --- ---\n")
    print(f"Final joke: {state['final_joke']}")
else:
    print("Joke failed quality gate - no punchline detected!")
