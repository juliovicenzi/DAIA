from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from models import get_chat_model

from daia.agents.tools import search_food_information


def get_daia_graph(checkpointer: PostgresSaver) -> CompiledGraph:
    llm = get_chat_model()
    tools = [search_food_information]

    graph = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        prompt="You are a helpful agent for nutrion information",
    )
    return graph
