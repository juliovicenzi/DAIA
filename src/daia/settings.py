"""settings object for the project, including secret env variables"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# all available models that support function calling
_function_calling_model = Literal[
    "mistral-7b-instruct-v03-fc",
    "llama-3-8b-instruct",
    "llama-3-1-8b-instruct",
    "llama-3-2-3b-instruct",
    "llama-3-3-70b-instruct",
]
_embedding_model = Literal["gte-large", "bge-m3"]


class DaiaSettings(BaseSettings):
    """Settings for the project, including secrets.
    All variables must be set via .env file or environment variables
    """

    OPENAI_API_KEY: SecretStr
    OPENAI_URL: str
    OPENAI_MODEL: _function_calling_model = "llama-3-3-70b-instruct"
    OPENAI_EMBEDDING_MODEL: _embedding_model = "gte-large"

    PG_URI: str
    PG_USER: str
    PG_PASSWORD: str
    PG_DATABASE: str
    PG_PORT: int
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_ignore_empty=True
    )


@lru_cache
def get_settings() -> DaiaSettings:
    """Singleton implementation of the settings object"""
    return DaiaSettings() # type: ignore
