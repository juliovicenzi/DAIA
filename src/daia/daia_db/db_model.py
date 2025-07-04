"""SQLModel for database querying and writing"""

from functools import lru_cache
from typing import Annotated, Any

from langgraph.checkpoint.postgres import PostgresSaver
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
from sqlalchemy import URL, Engine
from sqlmodel import Field, Index, SQLModel, create_engine

from daia.settings import get_settings


class FoodNutrientsBase(SQLModel):
    """Base class for FoodNutrients model.

    Represents information about a food considering a 100 gram portion.
    """

    food_name: str = Field(primary_key=True, unique=True, max_length=250)
    carbohydrates_g: float = Field(description="Carbohydrates in grams")
    energy_kcal: float = Field(description="Energy in kcal")
    lipid_g: float = Field(description="lipids (fats) in grams")
    protein_g: float = Field(description="proteins in grams")
    fiber_g: float = Field(description="fiber in grams")


class FoodNutrients(FoodNutrientsBase, table=True):
    """SQLModel for database querying and writing."""

    food_embedding: Any = Field(sa_type=Vector(1024), default=None)  # type: ignore

    __tablename__: str = "food_nutrients"

    # create HSNW index on food_embedding
    __table_args__ = (
        Index(
            "food_index",
            "food_embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"food_embedding": "vector_cosine_ops"},
        ),
    )


class FoodNutrientsSimilarity(FoodNutrientsBase):
    """Similarity model for FoodNutrients."""

    similarity: float = Field(
        description="Cosine similarity score, between 0 and 1, with 1 being the least similar."
    )


class FoodNutrientsPortionSize(FoodNutrientsBase):
    """Portion size model for FoodNutrients."""

    portion_size_g: Annotated[float, Field(description="Portion size in grams")]


class Meal(BaseModel):
    """Base class for Meal model.

    Represents information about a meal, as a list of foods,
    nutrional information and their portion sizes.
    """

    foods: Annotated[
        list[FoodNutrientsPortionSize], Field(description="Foods in the meal")
    ]


@lru_cache
def get_engine() -> Engine:
    """Singleton implementation of the database engine.

    A single engine should be used for the whole application,
    while multiple sessions can be used from it.

    Returns:
        The database engine
    """
    url = get_pg_url()
    engine = create_engine(url)
    return engine


@lru_cache
def get_pg_url() -> URL:
    """Singleton implementation of the database URL.

    Used to get access to the postgresql database
    for both similarity search and agent checkpoint.

    Returns:
        The database URL.

        Can be rendered as a string using the `render_as_string(hide_password=False)` method.
    """
    settings = get_settings()
    return URL.create(
        drivername="postgresql",
        username=settings.PG_USER,
        password=settings.PG_PASSWORD,
        host=settings.PG_URI,
        port=settings.PG_PORT,
        database=settings.PG_DATABASE,
    )


def setup_pg_checkpoint():
    """Setup the postgresql database for checkpointing.
    Must be called ONCE after the database is created.
    """
    url = get_pg_url().render_as_string(hide_password=False)
    with PostgresSaver.from_conn_string(url) as checkpoint:
        checkpoint.setup()
