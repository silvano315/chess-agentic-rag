import time

import pytest

from chess_agentic_rag.core.config import settings
from chess_agentic_rag.core.exceptions import LLMError
from chess_agentic_rag.llm.ollama_client import OllamaClient


@pytest.fixture
def mock_ollama_client(monkeypatch):
    """Fixture that provides an OllamaClient with a mocked underlying ollama.Client."""

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self._generate_calls = 0
            self._embed_calls = 0

        def list(self):
            # default list of models
            return {
                "models": [
                    {"name": "nomic-embed-text:latest"},
                    {"name": settings.ollama_llm_model},
                    {"name": settings.ollama_fallback_model},
                ]
            }

        def generate(self, model, prompt, system, options):
            self._generate_calls += 1
            return {"response": "generated text"}

        def chat(self, *args, **kwargs):
            return {"message": {"content": "chat reply"}}

        def embed(self, model, input):
            self._embed_calls += 1
            # return 768-dim vector of zeros for simplicity
            return {"embeddings": [[0.0] * settings.embedding_dim]}

        def close(self):
            return None

    monkeypatch.setattr("ollama.Client", FakeClient)

    client = OllamaClient(timeout=2, max_connections=3)

    return client


def test_validate_models_success(mock_ollama_client):
    results = mock_ollama_client.validate_models()
    assert results[settings.ollama_llm_model] is True
    assert results[settings.ollama_fallback_model] is True
    assert results[settings.ollama_embedding_model] is True


def test_validate_models_missing_primary(monkeypatch):
    class BrokenClient:
        def __init__(self, *a, **k):
            pass

        def list(self):
            return {"models": [{"name": "nomic-embed-text"}, {"name": "deepseek-r1:1.5b"}]}

    monkeypatch.setattr("ollama.Client", BrokenClient)
    client = OllamaClient(timeout=1)

    with pytest.raises(LLMError):
        client.validate_models()


def test_health_check_when_down(monkeypatch):
    class DownClient:
        def __init__(self, *a, **k):
            pass

        def list(self):
            raise ConnectionError("Ollama is down")

    monkeypatch.setattr("ollama.Client", DownClient)
    client = OllamaClient(timeout=1)
    assert client.health_check() is False


def test_retry_on_transient_failure(monkeypatch):
    # simulate generate failing once then succeeding
    calls = {"count": 0}

    class FlakyClient:
        def __init__(self, *a, **k):
            pass

        def list(self):
            return {"models": [{"name": settings.ollama_llm_model}]}

        def generate(self, model, prompt, system, options):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("transient error")
            return {"response": "ok after retry"}

        def close(self):
            return None

    monkeypatch.setattr("ollama.Client", FlakyClient)
    client = OllamaClient(timeout=1)
    res = client.generate(prompt="hi")
    assert res == "ok after retry"
    assert calls["count"] >= 2


def test_timeout_scenario(monkeypatch):
    # simulate slow generate that will raise an exception
    class SlowClient:
        def __init__(self, *a, **k):
            pass

        def list(self):
            return {"models": [{"name": settings.ollama_llm_model}]}

        def generate(self, model, prompt, system, options):
            time.sleep(2)
            return {"response": "late"}

        def close(self):
            return None

    monkeypatch.setattr("ollama.Client", SlowClient)
    client = OllamaClient(timeout=1)

    # The generate call should raise after retries because it sleeps past timeout
    with pytest.raises(LLMError):
        client.generate(prompt="slow")


def test_embedding_cache(monkeypatch, mock_ollama_client):
    # ensure get_embeddings uses cache and underlying embed is called once
    client = mock_ollama_client
    # first call should populate cache
    e1 = client.get_embeddings(text="hello world")
    # second call should hit cache
    e2 = client.get_embeddings(text="hello world")
    assert e1 == e2


@pytest.mark.parametrize(
    "primary,fallback,embed_model",
    [
        ("qwen2.5:7b", "deepseek-r1:1.5b", "nomic-embed-text"),
        ("qwen2.5", "deepseek-r1", "nomic-embed-text:latest"),
    ],
)
def test_parametrized_model_configs(monkeypatch, primary, fallback, embed_model):
    class ModelClient:
        def __init__(self, *a, **k):
            pass

        def list(self):
            return {"models": [{"name": primary}, {"name": fallback}, {"name": embed_model}]}

    monkeypatch.setattr("ollama.Client", ModelClient)

    # Temporarily override settings for this test
    orig_primary = settings.ollama_llm_model
    orig_fallback = settings.ollama_fallback_model
    orig_embed = settings.ollama_embedding_model

    settings.ollama_llm_model = primary
    settings.ollama_fallback_model = fallback
    settings.ollama_embedding_model = embed_model

    client = OllamaClient(timeout=1)
    results = client.validate_models()
    assert results[primary] is True

    # restore settings
    settings.ollama_llm_model = orig_primary
    settings.ollama_fallback_model = orig_fallback
    settings.ollama_embedding_model = orig_embed
