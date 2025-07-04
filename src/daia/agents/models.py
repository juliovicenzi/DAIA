"""Chat and embedding models for Daia"""

from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from openai import OpenAI
from pydantic import SecretStr

from daia.settings import get_settings


@lru_cache
def get_chat_model(temperature: float = 0.2) -> ChatOpenAI:
    """Singleton implementation of the OpenAI chat model

    Args:
        temperature: temperature of the model. Defaults to 0.2.
    Returns:
        ChatOpenAI: OpenAI chat model
    """
    settings = get_settings()
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=temperature,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        base_url=settings.OPENAI_URL,
    )


@lru_cache
def get_embedding_model() -> OpenAI:
    """Singleton implementation of the OpenAI client
    Used for embedding queries and documents

    We don't use langchain_openai.embeddings.OpenAIEmbeddings
    due to bugs found during testing.

    Returns:
        OpenAI: OpenAI client
    """
    settings = get_settings()
    model = OpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_URL,
    )
    return model


def embed_documents(documents: list[str]) -> list[list[float]]:
    """Embed documents using the OpenAI embedding model

    Args:
        documents: list of documents to embed
    Returns:
        A list of embeddings
    """
    model = get_embedding_model()
    settings = get_settings()
    # OpenAI returns a list[Embedding]
    embeddings = model.embeddings.create(
        model=settings.OPENAI_EMBEDDING_MODEL, input=documents
    ).data
    # Unpack it, to return a list[list[float]]
    return [emb.embedding for emb in embeddings]


def embed_query(query: str) -> list[float]:
    """Embed query using the OpenAI embedding model

    Args:
        query: query to embed
    Returns:
        The embedding
    """
    return embed_documents([query])[0]
