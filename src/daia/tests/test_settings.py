import pytest

from daia.settings import get_settings


def test_env_var():
    settings = get_settings()
    assert settings.OPENAI_API_KEY is not None


if __name__ == "__main__":
    pytest.main()
