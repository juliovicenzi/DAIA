"""Tools used by DAIA"""

from typing import Iterable

from langchain.tools import tool
from sqlmodel import Session, select

from daia.agents.models import get_embedding_model
from daia.daia_db.db_model import FoodNutrients, get_engine


@tool
def search_food_information(food_name: str) -> Iterable[FoodNutrients] | None:
    """Searches for food information in the neo4j database

    Args:
        food_name (str): The name of the food to search for
    Returns:
        list[FoodNutrients] | None: A list of FoodNutrients objects, None on failure
    """
    query_embedding = get_embedding_model().embed_query(food_name)
    try:
        with Session(get_engine()) as session:
            records = session.exec(
                select(FoodNutrients)
                .order_by(FoodNutrients.food_embedding.l2_distance(query_embedding))
                .limit(5)
            ).all()
    except Exception:
        return None
    return records
