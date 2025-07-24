from typing import Any

import fastapi
from fastapi.responses import RedirectResponse
from langchain_core.messages import BaseMessage

from daia.agents.agent import invoke_agent
from daia.agents.tools import search_food_information
from daia.daia_db.db_model import FoodNutrientsSimilarity

app = fastapi.FastAPI(title="DAIA")


@app.get("/", include_in_schema=False)
async def redirect_docs() -> RedirectResponse:
    return RedirectResponse("/docs")


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
def prompt_agent(prompt: str, thread_id: str) -> list[BaseMessage]:
    """Prompts the agent"""
    return invoke_agent(prompt, thread_id=thread_id)["messages"]
