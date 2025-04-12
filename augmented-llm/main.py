import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from pprint import pprint
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from config import Settings


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(None, description="Why this query is relevant to the user's request.")


def multiply(a: int, b: int) -> int:
    return a * b


def main() -> None:
    settings = Settings()

    llm = ChatOpenAI(model=settings.MODEL_NAME, api_key=settings.OPENAI_API_KEY)

    structured_llm = llm.with_structured_output(SearchQuery)
    output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")
    pprint(output)

    llm_with_tools = llm.bind_tools([multiply])
    msg = llm_with_tools.invoke("What is 2 times 3?")
    pprint(msg.tool_calls)


if __name__ == "__main__":
    main()
