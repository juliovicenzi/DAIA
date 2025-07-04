from daia.agents.models import get_chat_model, get_embedding_model


def test_get_chat_model():
    model = get_chat_model()
    model.invoke("hello")
