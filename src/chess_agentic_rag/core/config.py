from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via .env file or environment variables.
    See .env.example for all available options.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ============================================
    # LLM Configuration (Ollama)
    # ============================================

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )

    ollama_llm_model: str = Field(
        default="qwen2.5:7b",
        description="Primary LLM model for agent reasoning",
    )

    ollama_fallback_model: str = Field(
        default="deepseek-r1:1.5b",
        description="Fallback LLM model for complex queries",
    )

    ollama_embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model for semantic search",
    )

    ollama_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="LLM generation temperature",
    )

    ollama_max_tokens: int = Field(
        default=2000,
        gt=0,
        description="Maximum tokens to generate",
    )

    ollama_timeout: int = Field(
        default=120,
        gt=0,
        description="Request timeout in seconds",
    )

    # ============================================
    # Vector Store Configuration (ChromaDB)
    # ============================================

    chroma_persist_dir: str = Field(
        default="./data/vector_store/chroma_db",
        description="ChromaDB persistence directory",
    )

    chroma_collection_name: str = Field(
        default="chess_knowledge",
        description="ChromaDB collection name",
    )

    embedding_dim: int = Field(
        default=768,
        description="Embedding dimension (nomic-embed-text: 768)",
    )

    # ============================================
    # Data Pipeline Configuration
    # ============================================

    chunk_size: int = Field(
        default=600,
        gt=0,
        description="Text chunk size in tokens",
    )

    chunk_overlap: int = Field(
        default=100,
        ge=0,
        description="Chunk overlap in tokens",
    )

    data_raw_dir: str = Field(
        default="./data/raw",
        description="Raw data directory",
    )

    data_processed_dir: str = Field(
        default="./data/processed",
        description="Processed data directory",
    )

    # ============================================
    # API Configuration (FastAPI)
    # ============================================

    api_host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )

    api_port: int = Field(
        default=8000,
        gt=0,
        le=65535,
        description="API server port",
    )

    api_reload: bool = Field(
        default=True,
        description="Enable auto-reload in development",
    )

    api_cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins",
    )

    api_rate_limit: int = Field(
        default=60,
        gt=0,
        description="API rate limit (requests per minute)",
    )

    # ============================================
    # Logging Configuration
    # ============================================

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    log_file: str = Field(
        default="./logs/app.log",
        description="Log file path",
    )

    log_rotation: str = Field(
        default="500 MB",
        description="Log rotation threshold",
    )

    log_retention: str = Field(
        default="10 days",
        description="Log retention period",
    )

    # ============================================
    # External APIs (Optional)
    # ============================================

    lichess_api_token: str = Field(
        default="",
        description="Lichess API token for game data",
    )

    chess_com_api_base: str = Field(
        default="https://api.chess.com/pub",
        description="Chess.com public API base URL",
    )

    # ============================================
    # Tool Configuration
    # ============================================

    stockfish_path: str = Field(
        default="/usr/local/bin/stockfish",
        description="Stockfish engine path (M7 - optional)",
    )

    stockfish_depth: int = Field(
        default=20,
        gt=0,
        description="Stockfish evaluation depth",
    )

    stockfish_threads: int = Field(
        default=4,
        gt=0,
        description="Stockfish thread count",
    )

    # ============================================
    # Agent Configuration
    # ============================================

    agent_max_iterations: int = Field(
        default=5,
        gt=0,
        description="Maximum reasoning iterations before timeout",
    )

    enable_elo_fetcher: bool = Field(
        default=True,
        description="Enable ELO fetcher tool",
    )

    enable_pgn_parser: bool = Field(
        default=True,
        description="Enable PGN parser tool",
    )

    enable_game_search: bool = Field(
        default=True,
        description="Enable game search tool",
    )

    enable_stockfish: bool = Field(
        default=False,
        description="Enable Stockfish tool (requires installation)",
    )

    # ============================================
    # Development & Testing
    # ============================================

    environment: str = Field(
        default="development",
        description="Environment mode (development, production)",
    )

    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    test_data_dir: str = Field(
        default="./tests/fixtures",
        description="Test data directory",
    )


# Global settings instance
settings = Settings()
