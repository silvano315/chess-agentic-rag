from loguru import logger

from chess_agentic_rag.core.exceptions import LLMError
from chess_agentic_rag.llm.ollama_client import OllamaClient


def main() -> int:
    logger.info("Starting manual Ollama check")

    try:
        client = OllamaClient()
        logger.info("Ollama client initialized", base_url=client.base_url, model=client.model)

        logger.info("Running health check...")
        if not client.health_check():
            logger.error("Ollama health check failed. Is Ollama running (ollama serve)?")
            return 1

        logger.info("Listing models...")
        models = client.list_models()
        logger.info(f"Found {len(models)} models")
        for m in models:
            logger.info(f" - {m.get('name') or m.get('model') or m.get('id')}")

        logger.info("Validating required models...")
        try:
            client.validate_models()
            logger.info("Model validation succeeded")
        except LLMError as e:
            logger.error(f"Model validation failed: {e}")
            return 2

        logger.info("Testing generation...")
        prompt = "Say hello in one short sentence."
        try:
            resp = client.generate(prompt, max_tokens=30)
            logger.info(f"Generate result: {resp}")
        except LLMError as e:
            logger.error(f"Generation failed: {e}")
            return 3

        logger.info("Testing embeddings...")
        try:
            emb = client.get_embeddings("test embedding")
            logger.info(f"Embeddings dimension: {len(emb)}")
        except LLMError as e:
            logger.error(f"Embeddings failed: {e}")
            return 4

        client.close()
        logger.success("Manual Ollama check completed successfully")
        return 0

    except Exception as e:
        logger.exception("Unexpected error during manual Ollama check")
        return 99


if __name__ == "__main__":
    raise SystemExit(main())
