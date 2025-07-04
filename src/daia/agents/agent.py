"""DAIA Agent"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent

from daia.agents.models import get_chat_model
from daia.agents.tools import search_food_information
from daia.daia_db.db_model import get_pg_url


def get_daia_graph(checkpointer: PostgresSaver) -> CompiledGraph:
    """Get the DAIA graph

    Args:
        checkpointer: The checkpointer to use for the graph

    Returns:
        The DAIA graph
    """
    llm = get_chat_model()
    tools = [search_food_information]

    graph = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        prompt=SystemMessage("You are a helpful agent for nutrition information"),
    )
    return graph


def invoke_agent(prompt: str, thread_id: str) -> dict[str, Any]:
    """Invokes the agent with a prompt for a given thread

    Args:
        prompt: The prompt to send to the agent
        thread_id: The ID of the thread to invoke the agent

    Returns:
        The response from the agent
    """
    url = get_pg_url().render_as_string(hide_password=False)
    with PostgresSaver.from_conn_string(url) as checkpointer:
        graph = get_daia_graph(checkpointer)
        return graph.invoke(
            {"messages": [HumanMessage(prompt)]},
            config={"thread_id": thread_id},  # type: ignore
        )
