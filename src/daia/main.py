import fastapi

app = fastapi.FastAPI()


@app.get("/")
async def hello() -> dict[str, str]:
    return {"hello": "world"}


@app.get("/ping")
async def ping() -> dict[str, str]:
    return {"ping": "pong"}


@app.post("/v1/search_similarity")
def search_food_information(food_name: str) -> dict | None:
    return search_food_information(food_name)
