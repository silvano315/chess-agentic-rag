import threading
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any

import ollama
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

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
        max_connections: int = 10,
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
        self.max_connections = max_connections
        # Semaphore to limit concurrent requests and provide simple connection pooling
        self._semaphore = threading.BoundedSemaphore(value=self.max_connections)

        try:
            # Initialize underlying Ollama client with configured timeout
            self.client = ollama.Client(host=self.base_url, timeout=self.timeout)
            logger.bind(component="ollama_client").info(
                "Ollama client initialized",
                base_url=self.base_url,
                model=self.model,
                timeout=self.timeout,
                max_connections=self.max_connections,
            )
        except Exception as e:
            logger.bind(component="ollama_client").exception(
                "Failed to initialize Ollama client",
                error=str(e),
            )
            raise LLMError(f"Failed to initialize Ollama client: {e}") from e

    def _acquire_connection(self, acquire_timeout: int | None = None) -> None:
        """
        Acquire a semaphore slot for a connection.

        Args:
            acquire_timeout: Seconds to wait for a free slot. Defaults to client's timeout.

        Raises:
            LLMError: If a connection slot cannot be acquired within the timeout.
        """
        timeout_sec = acquire_timeout or self.timeout
        acquired = self._semaphore.acquire(timeout=timeout_sec)
        if not acquired:
            logger.bind(component="ollama_client").error(
                "Connection pool exhausted",
                timeout=timeout_sec,
                max_connections=self.max_connections,
            )
            raise LLMError(
                f"Timed out waiting for an Ollama connection slot after {timeout_sec}s"
            )

    @contextmanager
    def _connection_slot(self, acquire_timeout: int | None = None) -> Generator[None, None, None]:
        """Context manager that acquires/releases a connection slot.

        Yields:
            None
        """
        try:
            self._acquire_connection(acquire_timeout=acquire_timeout)
            yield
        finally:
            try:
                self._semaphore.release()
            except ValueError:
                # Ignore over-release attempts but log for visibility
                logger.bind(component="ollama_client").warning(
                    "Semaphore release failed - possibly double release"
                )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
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

        if system is None:
            system = ""

        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            **kwargs,
        }

        bound_logger = logger.bind(component="ollama_client", model=model)

        with self._connection_slot(acquire_timeout=self.timeout):
            try:
                bound_logger.debug(
                    "Generating completion",
                    prompt_length=len(prompt),
                    temperature=temperature,
                )

                response = self.client.generate(
                    model=model,
                    prompt=prompt,
                    system=system,
                    options=options,
                )

                generated_text = response.get("response")

                bound_logger.debug(
                    "Generation complete",
                    response_length=len(generated_text) if generated_text else 0,
                )

                if not generated_text:
                    raise LLMError("Ollama returned an empty generation response")

                return str(generated_text)

            except LLMError:
                raise
            except Exception as e:
                bound_logger.bind(prompt_preview=prompt[:120]).exception(
                    "Generation failed due to exception",
                    error=str(e),
                )
                raise LLMError(f"Failed to generate completion: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
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

        options = {"temperature": temperature, **kwargs}

        bound_logger = logger.bind(component="ollama_client", model=model)

        with self._connection_slot(acquire_timeout=self.timeout):
            try:
                bound_logger.debug(
                    "Starting chat completion",
                    num_messages=len(messages),
                    has_tools=tools is not None,
                )

                response = self.client.chat(
                    model=model,
                    messages=messages,
                    tools=tools,
                    options=options,
                )

                bound_logger.debug(
                    "Chat completion successful",
                    has_tool_calls="tool_calls" in response.get("message", {}),
                )

                return dict(response)

            except LLMError:
                raise
            except Exception as e:
                bound_logger.exception(
                    "Chat completion failed",
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

        options = {"temperature": temperature, **kwargs}

        bound_logger = logger.bind(component="ollama_client", model=model)

        with self._connection_slot(acquire_timeout=self.timeout):
            try:
                bound_logger.debug(
                    "Starting streaming chat",
                    num_messages=len(messages),
                )

                stream = self.client.chat(
                    model=model,
                    messages=messages,
                    stream=True,
                    options=options,
                )

                for chunk in stream:
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]

                bound_logger.debug("Streaming complete")

            except LLMError:
                raise
            except Exception as e:
                bound_logger.exception(
                    "Streaming failed",
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

        bound_logger = logger.bind(component="ollama_client", model=model)

        with self._connection_slot(acquire_timeout=self.timeout):
            try:
                bound_logger.debug("Generating embeddings", text_length=len(text))

                response = self.client.embed(model=model, input=text)
                embeddings = response["embeddings"][0]

                bound_logger.debug(
                    "Embeddings generated",
                    embedding_dim=len(embeddings),
                )

                return list(embeddings)

            except Exception as e:
                bound_logger.exception("Embedding generation failed", error=str(e))
                raise LLMError(f"Failed to generate embeddings: {e}") from e

    def list_models(self) -> list[dict[str, Any]]:
        """
        List all available models in Ollama.

        Returns:
            List of model information dictionaries

        Raises:
            LLMError: If listing models fails
        """
        bound_logger = logger.bind(component="ollama_client")

        with self._connection_slot(acquire_timeout=self.timeout):
            try:
                response = self.client.list()
                models = response.get("models", [])

                bound_logger.info("Listed available models", count=len(models))
                return list(models)

            except Exception as e:
                bound_logger.exception("Failed to list models", error=str(e))
                raise LLMError(f"Failed to list models: {e}") from e

    def validate_models(self, acquire_timeout: int | None = None) -> dict[str, bool]:
        """
        Validate that required Ollama models are available.

        Checks the presence of the primary LLM model, fallback model,
        and embedding model as defined in `settings`.

        Args:
            acquire_timeout: Seconds to wait for a connection slot. Defaults to client's timeout.

        Returns:
            A mapping of model_name -> bool indicating availability.

        Raises:
            LLMError: If the primary model (`settings.ollama_llm_model`) is missing
                      or if the Ollama API call fails.
        """
        primary = settings.ollama_llm_model
        fallback = settings.ollama_fallback_model
        embed = settings.ollama_embedding_model

        required_models = [primary, fallback, embed]

        bound_logger = logger.bind(component="ollama_client")

        # Use a single slot for the listing operation
        with self._connection_slot(acquire_timeout=acquire_timeout or self.timeout):
            try:
                resp = self.client.list()
            except Exception as e:
                bound_logger.exception("Failed to list models during validation", error=str(e))
                raise LLMError(f"Failed to validate models: {e}") from e

        available_raw = resp.get("models", [])

        # Normalize model names from returned dicts
        model_names: list[str] = []
        for m in available_raw:
            if not isinstance(m, dict):
                continue
            name = m.get("name") or m.get("model") or m.get("id") or ""
            if name:
                model_names.append(str(name))

        results: dict[str, bool] = {}
        missing: list[str] = []

        for req in required_models:
            found = any(
                (req in mn) or mn.startswith(req.split(":")[0])
                for mn in model_names
            )
            results[req] = found
            if not found:
                missing.append(req)

        if missing:
            bound_logger.warning(
                "Missing required Ollama models",
                missing=missing,
                suggestion="Run: bash scripts/setup_ollama.sh or set OLLAMA_BASE_URL/OLLAMA_* env vars",
            )

        # Primary model missing is a fatal error for operations that rely on it
        if not results.get(primary, False):
            msg = (
                f"Primary model '{primary}' is not available in Ollama. "
                "Install or configure the model before proceeding. "
                "Run: bash scripts/setup_ollama.sh to install recommended models."
            )
            bound_logger.error("Primary model missing", primary=primary)
            raise LLMError(msg)

        bound_logger.info("Model validation complete", checked=len(required_models))
        return results

    def health_check(self) -> bool:
        """
        Check if Ollama service is healthy.

        Returns:
            True if service is responding, False otherwise
        """
        bound_logger = logger.bind(component="ollama_client")
        try:
            with self._connection_slot(acquire_timeout=5):
                self.client.list()
            bound_logger.info("Health check passed", base_url=self.base_url)
            return True
        except Exception as e:
            bound_logger.warning("Health check failed", base_url=self.base_url, error=str(e))
            return False

    def close(self) -> None:
        """
        Close any managed resources (clients/pools).

        This is safe to call multiple times.
        """
        try:
            # If underlying client exposes a close method, call it.
            close_fn = getattr(self.client, "close", None)
            if callable(close_fn):
                close_fn()
            logger.bind(component="ollama_client").info("Closed Ollama client resources")
        except Exception as e:
            logger.bind(component="ollama_client").exception("Error closing Ollama client", error=str(e))
