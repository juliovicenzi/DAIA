"""SQLModel for database querying and writing"""

from functools import lru_cache
from typing import Any

import pandas as pd
from pgvector.sqlalchemy import Vector
from sqlalchemy import URL, Engine
from sqlmodel import Field, Index, SQLModel, create_engine

from daia.agents.models import get_embedding_model
from daia.settings import get_settings


class FoodNutrients(SQLModel, table=True):
    """SQLModel for database querying and writing"""

    food_name: str = Field(primary_key=True, unique=True, max_length=250)
    food_embedding: Any = Field(sa_type=Vector(1024), default=None)  # type: ignore
    carbohydrates_g: float
    energy_kcal: float
    lipid_g: float
    protein_g: float
    fiber_g: float

    __tablename__: str = "food_nutrients"

    # create HSNW index on food_embedding
    __table_args__ = (
        Index(
            "food_index",
            "food_embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"food_embedding": "vector_l2_ops"},
        ),
    )


@lru_cache
def get_engine() -> Engine:
    """Singleton implementation of the database engine
    A single engine should be used for the whole application,
    while multiple sessions can be used from it.
    """
    settings = get_settings()
    url = URL.create(
        drivername="postgresql",
        username=settings.PG_USER,
        password=settings.PG_PASSWORD,
        host=settings.PG_URI,
        port=settings.PG_PORT,
        database=settings.PG_DATABASE,
    )
    engine = create_engine(url)
    return engine
