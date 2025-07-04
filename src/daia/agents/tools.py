"""Tools used by DAIA"""

import math

from langchain.tools import tool
from sqlmodel import Session, select

from daia.agents.models import embed_query
from daia.daia_db.db_model import (
    FoodNutrients,
    FoodNutrientsSimilarity,
    Meal,
    get_engine,
)


@tool
def search_food_information(
    food_name: str, result_limit: int = 5, minimum_similarity: float = 0.15
) -> list[FoodNutrientsSimilarity]:
    """Searches for food information in the Postgres database, in 100 gram portions.

    Cosine similarity is in range [0,1], the closer to zero, greater the similarity.

    Args:
        food_name: The name of the food to search for
        result_limit: The maximum number of results to return. Defaults to 5.
        minimum_similarity: Only return results with similarity lower than this. Defaults to 0.15.
    Returns:
        A list the food informationa and similarity score, or an empty list if not found.
    """
    query_embedding = embed_query(food_name)
    cosine_distance = FoodNutrients.food_embedding.cosine_distance(query_embedding)

    # don't catch any expections, as the agent should handle them
    with Session(get_engine()) as session:
        records = session.exec(
            select(FoodNutrients, cosine_distance)
            .where(cosine_distance < minimum_similarity)
            .order_by(cosine_distance)
            .limit(result_limit)
        ).all()
    return [
        FoodNutrientsSimilarity(similarity=similarity, **food_nutrients.model_dump())
        for food_nutrients, similarity in records
    ]


# TODO: add structured response for bolus output
# TODO: add structured response for correction output
# TODO: add the idea of a user, alongside its information, such as insulin_to_carb_ratio, target_blood_glucose, etc


@tool
def meal_bolus_calculator(meal: Meal, insulin_to_carb_ratio: float) -> int:
    """Used to calculate the amount of insulin units required to bolus for a meal
    based on the nutrional information of the foods, and their respective quantities

    Args:
        meal: The meal comprised of foods with their nutrional information and portions
        insulin_to_carb_ratio: The ratio of units of insulin to units of carbohydrates

    returns:
        The number of units of insulin to bolus
    Returns:
        A breakdown of the bolus calcualtion
    """
    total_carbohydrates = sum(
        (food.carbohydrates_g * food.portion_size_g / 100 for food in meal.foods)
    )

    units = math.floor(total_carbohydrates / insulin_to_carb_ratio)
    return units


@tool
def correction_bolus_calculator(
    current_blood_glucose: float,
    target_blood_glucose: float,
    insulin_sensitivity: float,
) -> int:
    """Calculate the number of units of insulin to take to correct the blood glucose level.

    If the target is lower than the current,
    the function returns a negative value of units to remove from a meal bolus.

    If the calcualtion is not tied to a meal bolus calculation,
    the user should then be instructed to have a meal
    to compensate for the low blood glucose.

    Args:
        current_blood_glucose: current blood glucose level
        target_blood_glucose: target blood glucose level
        insulin_sensitivity: the user's sensitivity to insulin,
            the amound of blood glucose a unit of insulin decreases, in mg/dL units
    Returns:
        The number of units of insulin to take to correct the blood glucose level.
        A negative value indicates less units of insulin to take, or a meal recommendation.
    """
    return math.floor(
        (target_blood_glucose - current_blood_glucose) / insulin_sensitivity
    )
