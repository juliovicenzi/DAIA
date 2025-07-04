from daia.agents.tools import search_food_information


def test_similarity():
    records = search_food_information.invoke("Oat")
    print(records)
