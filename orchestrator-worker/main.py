import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import operator
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph

from config import Settings


config = Settings()
llm = ChatOpenAI(model=config.MODEL_NAME, api_key=config.OPENAI_API_KEY)


class Section(BaseModel):
    name: str = Field(description="Name for this section of the report")
    description: str = Field(description="Brief overview of the main topics and concepts to be covered in this section.")


class Sections(BaseModel):
    sections: list[Section] = Field(description="Sections of the report.")


planner = llm.with_structured_output(Sections)


class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]
    final_report: str


class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


def orchestrator(state: State):
    """Orchestrator that generates a plan for the report"""

    report_sections = planner.invoke(
        [SystemMessage(content="Generate a plan for the report."), HumanMessage(content=f"Here is the report topic: {state['topic']}")]
    )

    return {"sections": report_sections.sections}


def llm_call(state: WorkerState) -> dict:
    """Worker writes a section of the report"""

    section = llm.invoke(
        [
            SystemMessage(
                content="Write a report section following the provided name and description. Include no preamble for eache section. Use markdown formatting."
            ),
            HumanMessage(content=f"Here is the section name: {state['section'].name} and description: {state['section'].description}"),
        ]
    )

    return {"completed_sections": [section.content]}


def synthesizer(state: State) -> dict:
    """Synthesize full report from sections"""

    completed_sections = state["completed_sections"]
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}


def assign_workers(state: State):
    return [Send("llm_call", {"section": s}) for s in state["sections"]]


orchestrator_worker_builder = StateGraph(State)

orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges("orchestrator", assign_workers, ["llm_call"])
orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

orchestrator_worker = orchestrator_worker_builder.compile()

state = orchestrator_worker.invoke({"topic": "Create a report on LLM scaling laws"})