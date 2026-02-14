import pytest
from loguru import logger

from chess_agentic_rag.core.config import settings
from chess_agentic_rag.core.exceptions import LLMError
from chess_agentic_rag.llm.ollama_client import OllamaClient


class TestOllamaConnection:
    """Test suite for Ollama client connection and setup."""

    @pytest.fixture
    def client(self) -> OllamaClient:
        """Create Ollama client instance for testing."""
        return OllamaClient()

    def test_client_initialization(self, client: OllamaClient) -> None:
        """Test that Ollama client can be initialized."""
        assert client is not None
        assert client.base_url == settings.ollama_base_url
        assert client.model == settings.ollama_llm_model
        logger.info("✅ Ollama client initialized successfully")

    @pytest.mark.requires_ollama
    def test_health_check(self, client: OllamaClient) -> None:
        """Test that Ollama service is running and responding."""
        is_healthy = client.health_check()
        assert is_healthy is True, "Ollama service is not responding"
        logger.info("✅ Ollama service health check passed")

    @pytest.mark.requires_ollama
    def test_list_models(self, client: OllamaClient) -> None:
        """Test that required models are available."""
        models = client.list_models()
        assert len(models) > 0, "No models found in Ollama"

        model_names = [model.get("name", "") for model in models]
        logger.info(f"Available models: {model_names}")

        # Check for required models
        required_models = [
            "deepseek-r1:1.5b",
            "qwen2.5:7b",
            "nomic-embed-text",
        ]

        missing_models = []
        for required_model in required_models:
            # Check if model exists (exact match or with tag)
            found = any(
                required_model in model_name or model_name.startswith(required_model.split(":")[0])
                for model_name in model_names
            )
            if not found:
                missing_models.append(required_model)

        if missing_models:
            logger.warning(f"⚠️  Missing recommended models: {missing_models}")
            logger.info("Run: bash scripts/setup_ollama.sh to install them")
        else:
            logger.info("✅ All required models are available")

    @pytest.mark.requires_ollama
    def test_simple_generation(self, client: OllamaClient) -> None:
        """Test basic text generation with a simple chess query."""
        prompt = "What is the Italian Opening in chess? Answer in one sentence."

        try:
            response = client.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=100,
            )

            assert response is not None
            assert len(response) > 0
            assert isinstance(response, str)

            logger.info("✅ Simple generation test passed")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Response: {response[:200]}...")  # First 200 chars

        except LLMError as e:
            pytest.fail(f"Generation failed: {e}")

    @pytest.mark.requires_ollama
    def test_chat_completion(self, client: OllamaClient) -> None:
        """Test chat completion with message history."""
        messages = [
            {
                "role": "system",
                "content": "You are a chess expert. Answer concisely.",
            },
            {
                "role": "user",
                "content": "What is the Sicilian Defense?",
            },
        ]

        try:
            response = client.chat(messages=messages, temperature=0.7)

            assert response is not None
            assert "message" in response
            assert "content" in response["message"]

            content = response["message"]["content"]
            assert len(content) > 0

            logger.info("✅ Chat completion test passed")
            logger.info(f"Response: {content[:200]}...")

        except LLMError as e:
            pytest.fail(f"Chat completion failed: {e}")

    @pytest.mark.requires_ollama
    def test_streaming_chat(self, client: OllamaClient) -> None:
        """Test streaming chat completion."""
        messages = [
            {
                "role": "user",
                "content": "List 3 famous chess openings in a comma-separated format.",
            }
        ]

        try:
            chunks = list(client.stream_chat(messages=messages, temperature=0.7))

            assert len(chunks) > 0, "No chunks received from streaming"

            full_response = "".join(chunks)
            assert len(full_response) > 0

            logger.info("✅ Streaming chat test passed")
            logger.info(f"Received {len(chunks)} chunks")
            logger.info(f"Full response: {full_response[:200]}...")

        except LLMError as e:
            pytest.fail(f"Streaming chat failed: {e}")

    @pytest.mark.requires_ollama
    def test_embeddings_generation(self, client: OllamaClient) -> None:
        """Test embedding generation for semantic search."""
        text = "The Sicilian Defense is a popular chess opening"

        try:
            embeddings = client.get_embeddings(text=text)

            assert embeddings is not None
            assert isinstance(embeddings, list)
            assert len(embeddings) > 0

            # nomic-embed-text produces 768-dimensional embeddings
            assert len(embeddings) == 768, f"Expected 768 dimensions, got {len(embeddings)}"

            # Check that embeddings are floats
            assert all(isinstance(e, float) for e in embeddings)

            logger.info("✅ Embeddings generation test passed")
            logger.info(f"Text: {text}")
            logger.info(f"Embedding dimensions: {len(embeddings)}")
            logger.info(f"First 5 values: {embeddings[:5]}")

        except LLMError as e:
            pytest.fail(f"Embedding generation failed: {e}")

    @pytest.mark.requires_ollama
    def test_different_models(self, client: OllamaClient) -> None:
        """Test that different models can be used."""
        models_to_test = ["deepseek-r1:1.5b", "qwen2.5:7b"]

        prompt = "What is chess? You must provide only the final answer. Do not expose reasoning."

        for model in models_to_test:
            try:
                response = client.generate(
                    prompt=prompt,
                    model=model,
                    max_tokens=50,
                )

                assert response is not None
                assert len(response) > 0

                logger.info(f"✅ Model {model} test passed")
                logger.info(f"Response: {response}")

            except LLMError as e:
                logger.warning(f"⚠️  Model {model} not available or failed: {e}")


if __name__ == "__main__":
    """
    Run this script directly to quickly test Ollama connection:

        uv run python tests/unit/test_ollama_connection.py
    """
    import sys

    logger.info("=" * 60)
    logger.info("Ollama Connection Test - M0.3")
    logger.info("=" * 60)
    logger.info("")

    try:
        # Initialize client
        logger.info("1. Initializing Ollama client...")
        client = OllamaClient()
        logger.success("   ✅ Client initialized")
        logger.info("")

        # Health check
        logger.info("2. Checking Ollama service health...")
        if client.health_check():
            logger.success("   ✅ Ollama service is running")
        else:
            logger.error("   ❌ Ollama service is not responding")
            logger.info("   💡 Make sure Ollama is running: ollama serve")
            sys.exit(1)
        logger.info("")

        # List models
        logger.info("3. Listing available models...")
        models = client.list_models()
        if models:
            logger.success(f"   ✅ Found {len(models)} models:")
            for model in models:
                logger.info(f"      - {model.get('model', 'unknown')}")
        else:
            logger.error("   ❌ No models found")
            logger.info("   💡 Run: bash scripts/setup_ollama.sh")
            sys.exit(1)
        logger.info("")

        # Test generation
        logger.info("4. Testing text generation...")
        prompt = """
        You must answer directly.
        Do not include reasoning.
        Answer in exactly one sentence.

        This is a test prompt. Answer in one sentence if everything is fine.
        """
        response = client.generate(prompt, num_predict=300)
        logger.success("   ✅ Generation successful")
        logger.info(f"   Prompt: {prompt}")
        logger.info(f"   Response: {response[:150]}...")
        logger.info("")

        # Test embeddings
        logger.info("5. Testing embedding generation...")
        text = "The Sicilian Defense"
        embeddings = client.get_embeddings(text)
        logger.success(f"   ✅ Embeddings generated ({len(embeddings)} dimensions)")
        logger.info(f"   Text: {text}")
        logger.info(f"   First 5 values: {embeddings[:5]}")
        logger.info("")

        # Success summary
        logger.info("=" * 60)
        logger.success("🎉 All tests passed! Ollama is ready to use.")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Copy .env.example to .env (if not done)")
        logger.info("  2. Start working on M1: Data Pipeline")
        logger.info("  3. Run full test suite: uv run pytest")
        logger.info("")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.info("")
        logger.info("Troubleshooting:")
        logger.info("  1. Ensure Ollama is running: ollama serve")
        logger.info("  2. Check models are installed: ollama list")
        logger.info("  3. Install models: bash scripts/setup_ollama.sh")
        logger.info("  4. Check Ollama URL in .env: OLLAMA_BASE_URL")
        sys.exit(1)
