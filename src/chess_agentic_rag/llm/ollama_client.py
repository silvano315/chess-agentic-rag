from collections.abc import Iterator
from typing import Any

import ollama
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from chess_agentic_rag.core.config import settings
from chess_agentic_rag.core.exceptions import LLMError


class OllamaClient:
    """
    Wrapper for Ollama API with error handling and retries.

    This client provides methods for text generation, chat completion,
    and streaming responses using local Ollama models.

    Attributes:
        base_url: Ollama API base URL
        model: Default LLM model name
        client: Underlying Ollama client instance

    Example:
        >>> client = OllamaClient(model="deepseek-r1:1.5b")
        >>> response = client.generate("Explain the Sicilian Defense")
        >>> print(response)
    """

    def __init__(
            self,
            base_url: str | None = None,
            model: str | None = None,
            timeout: int | None = None,
    ) -> None:
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API URL. Defaults to settings value.
            model: Default model name. Defaults to settings value.
            timeout: Request timeout in seconds. Defaults to settings value.
        """
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_llm_model
        self.timeout = timeout or settings.ollama_timeout

        try:
            self.client = ollama.Client(host=self.base_url, timeout=self.timeout)
            logger.info(
                "Ollama client initialized",
                base_url=self.base_url,
                model=self.model,
            )
        except Exception as e:
            logger.error("Failed to initialize Ollama client", error=str(e))
            raise LLMError(f"Failed to initialize Ollama client: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text completion from prompt.

        Args:
            prompt: The input prompt text
            system: Optional system message for context
            model: Model to use. Defaults to client's default model.
            temperature: Sampling temperature (0.0-1.0). Defaults to settings.
            max_tokens: Maximum tokens to generate. Defaults to settings.
            **kwargs: Additional parameters passed to Ollama

        Returns:
            Generated text response

        Raises:
            LLMError: If generation fails after retries

        Example:
            >>> response = client.generate(
            ...     "What is the Italian Opening?",
            ...     system="You are a chess expert.",
            ...     temperature=0.7
            ... )
        """
        model = model or self.model
        temperature = temperature or settings.ollama_temperature
        max_tokens = max_tokens or settings.ollama_max_tokens

        try:
            logger.debug(
                "Generating completion",
                model=model,
                prompt_length=len(prompt),
                temperature=temperature,
            )

            options = {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs,
            }

            response = self.client.generate(
                model=model,
                prompt=prompt,
                system=system,
                options=options,
            )

            generated_text = response["response"]

            logger.debug(
                "Generation complete",
                model=model,
                response_length=len(generated_text),
            )

            return generated_text
        except Exception as e:
            logger.error(
                "Generation failed",
                model=model,
                error=str(e),
            )
            raise LLMError(f"Failed to generate completion: {e}") from e

    @retry(
       stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute chat completion with message history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            model: Model to use. Defaults to client's default model.
            tools: Optional list of tool/function definitions for function calling
            temperature: Sampling temperature. Defaults to settings.
            **kwargs: Additional parameters passed to Ollama

        Returns:
            Complete response dict including message and metadata

        Raises:
            LLMError: If chat completion fails after retries

        Example:
            >>> messages = [
            ...     {"role": "user", "content": "What is chess?"}
            ... ]
            >>> response = client.chat(messages)
            >>> print(response["message"]["content"])
        """
        model = model or self.model
        temperature = temperature or settings.ollama_temperature

        try:
            logger.debug(
                "Starting chat completion",
                model=model,
                num_messages=len(messages),
                has_tools=tools is not None,
            )

            options = {
                "temperature": temperature,
                **kwargs,
            }

            response = self.client.chat(
                model=model,
                messages=messages,
                tools=tools,
                options=options,
            )

            logger.debug(
                "Chat completion successful",
                model=model,
                has_tool_calls="tool_calls" in response.get("message", {}),
            )

            return response

        except Exception as e:
            logger.error(
                "Chat completion failed",
                model=model,
                error=str(e),
            )
            raise LLMError(f"Failed to complete chat: {e}") from e

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream chat completion responses.

        Args:
            messages: List of message dicts
            model: Model to use. Defaults to client's default model.
            temperature: Sampling temperature. Defaults to settings.
            **kwargs: Additional parameters passed to Ollama

        Yields:
            Text chunks as they are generated

        Raises:
            LLMError: If streaming fails

        Example:
            >>> messages = [{"role": "user", "content": "Explain the Ruy Lopez"}]
            >>> for chunk in client.stream_chat(messages):
            ...     print(chunk, end="", flush=True)
        """
        model = model or self.model
        temperature = temperature or settings.ollama_temperature

        try:
            logger.debug(
                "Starting streaming chat",
                model=model,
                num_messages=len(messages),
            )

            options = {
                "temperature": temperature,
                **kwargs,
            }

            stream = self.client.chat(
                model=model,
                messages=messages,
                stream=True,
                options=options,
            )

            for chunk in stream:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

            logger.debug("Streaming complete", model=model)

        except Exception as e:
            logger.error(
                "Streaming failed",
                model=model,
                error=str(e),
            )
            raise LLMError(f"Failed to stream chat: {e}") from e

    def get_embeddings(
        self,
        text: str,
        model: str | None = None,
    ) -> list[float]:
        """
        Generate embeddings for text.

        Args:
            text: Input text to embed
            model: Embedding model to use. Defaults to settings value.

        Returns:
            Embedding vector as list of floats

        Raises:
            LLMError: If embedding generation fails

        Example:
            >>> embeddings = client.get_embeddings("chess opening theory")
            >>> len(embeddings)  # Should be 768 for nomic-embed-text
            768
        """
        model = model or settings.ollama_embedding_model

        try:
            logger.debug("Generating embeddings", model=model, text_length=len(text))

            response = self.client.embed(model=model, input=text)
            embeddings = response["embeddings"][0]

            logger.debug(
                "Embeddings generated",
                model=model,
                embedding_dim=len(embeddings),
            )

            return embeddings

        except Exception as e:
            logger.error("Embedding generation failed", model=model, error=str(e))
            raise LLMError(f"Failed to generate embeddings: {e}") from e

    def list_models(self) -> list[dict[str, Any]]:
        """
        List all available models in Ollama.

        Returns:
            List of model information dictionaries

        Raises:
            LLMError: If listing models fails
        """
        try:
            response = self.client.list()
            models = response.get("models", [])

            logger.info("Listed available models", count=len(models))
            return models

        except Exception as e:
            logger.error("Failed to list models", error=str(e))
            raise LLMError(f"Failed to list models: {e}") from e

    def health_check(self) -> bool:
        """
        Check if Ollama service is healthy.

        Returns:
            True if service is responding, False otherwise
        """
        try:
            self.client.list()
            logger.info("Health check passed", base_url=self.base_url)
            return True
        except Exception as e:
            logger.warning("Health check failed", base_url=self.base_url, error=str(e))
            return False
