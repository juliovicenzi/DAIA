from sqlmodel import Session, select

from daia.agents.models import embed_query
from daia.daia_db.db_model import FoodNutrients, get_engine


def test_get_embedding_model():
    emb = embed_query("Ham")

    with Session(get_engine()) as session:
        result = session.exec(
            select(FoodNutrients)
            .order_by(FoodNutrients.food_embedding.l2_distance(emb))
            .limit(5)
        ).all()

        assert len(result) == 5
