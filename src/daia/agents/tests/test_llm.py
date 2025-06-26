from daia.agents.models import get_chat_model, get_embedding_model


def test_get_chat_model():
    model = get_chat_model()
    model.invoke("hello")


def test_get_embedding_model():
    model = get_embedding_model()
    model.embed_documents(["hello"])
