from typing import Any

import fastapi
from langchain_core.messages import BaseMessage

from daia.agents.agent import get_daia_graph, invoke_agent
from daia.agents.tools import search_food_information
from daia.daia_db.db_model import FoodNutrientsSimilarity

app = fastapi.FastAPI(title="DAIA")


@app.get("/")
async def hello() -> dict[str, str]:
    return {"hello": "world"}


@app.get("/v1/search_similarity", response_model=list[FoodNutrientsSimilarity])
def food_search(food_name: str) -> Any:
    """Searches for food information in the Postgres database
    based on cosine similarity

    Args:
        food_name (str): The name of the food to search for

    Returns:
        list[FoodNutritionSimilarity]: A list the food information and similarity score,
        or an empty list if not found
    """
    reponse = search_food_information.invoke(food_name)
    return reponse


@app.get("/v1/agent")
def prompt_agent(prompt: str, thread_id: int) -> list[BaseMessage]:
    """Prompts the agent with a prompt"""
    return invoke_agent(prompt, thread_id=thread_id)["messages"]
