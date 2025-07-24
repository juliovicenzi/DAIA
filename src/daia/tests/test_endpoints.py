import pytest
from fastapi.testclient import TestClient

from daia.main import app


@pytest.fixture
def test_client():
    return TestClient(app)


def test_search_vector(test_client: TestClient):
    response = test_client.get(
        "/v1/search_similarity", params={"food_name": "test_food"}
    )
    assert response.status_code == 200
    json_reponse = response.json()
    assert len(json_reponse) <= 5


def test_very_large_food_name(test_client: TestClient):
    with pytest.raises(ValueError):
        response = test_client.get(
            "/v1/search_similarity", params={"food_name": "test_food" * 1000}
        )
