"""Chat and embedding models for Daia"""

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from daia.settings import get_settings


@lru_cache
def get_chat_model(temperature: float = 0.2) -> ChatOpenAI:
    """Singleton implementation of the OpenAI chat model"""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=temperature,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_URL,
    )


@lru_cache
def get_embedding_model() -> OpenAIEmbeddings:
    """Singleton implementation of the OpenAI embedding model"""
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_URL,
        # maximum chunk size for this model
        chunk_size=32,
    )
