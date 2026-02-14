class ChessRAGException(Exception):  # noqa: N818
    """
    Base exception for all Chess RAG errors.

    All custom exceptions in this project inherit from this class,
    allowing for catch-all error handling when needed.
    """

    pass


class ConfigurationError(ChessRAGException):
    """Raised when there's an error in configuration."""

    pass


class DataPipelineError(ChessRAGException):
    """
    Errors during data loading, processing, or validation.

    This includes errors from loaders, processors, and the pipeline orchestrator.
    """

    pass


class LoaderError(DataPipelineError):
    """Errors when loading data from sources."""

    pass


class ProcessorError(DataPipelineError):
    """Errors when processing or transforming data."""

    pass


class RetrievalError(ChessRAGException):
    """
    Errors during document retrieval or vector search.

    This includes vector store errors, embedding errors, and query errors.
    """

    pass


class VectorStoreError(RetrievalError):
    """Errors related to vector database operations."""

    pass


class EmbeddingError(RetrievalError):
    """Errors when generating embeddings."""

    pass


class ToolExecutionError(ChessRAGException):
    """
    Errors during tool execution.

    This includes tool-specific errors (API failures, parsing errors, etc.)
    """

    pass


class ELOFetcherError(ToolExecutionError):
    """Errors when fetching ELO ratings."""

    pass


class PGNParserError(ToolExecutionError):
    """Errors when parsing PGN notation."""

    pass


class GameSearchError(ToolExecutionError):
    """Errors when searching for games."""

    pass


class StockfishError(ToolExecutionError):
    """Errors when interacting with Stockfish engine."""

    pass


class AgentError(ChessRAGException):
    """
    Errors in agent orchestration and reasoning.

    This includes planning errors, execution errors, and reasoning failures.
    """

    pass


class PlanningError(AgentError):
    """Errors during query planning and decomposition."""

    pass


class ExecutionError(AgentError):
    """Errors during action execution."""

    pass


class ReasoningError(AgentError):
    """Errors in multi-step reasoning process."""

    pass


class MemoryError(ChessRAGException):
    """
    Errors in memory management.

    This includes conversation memory and working memory errors.
    """

    pass


class LLMError(ChessRAGException):
    """
    Errors when interacting with LLM backend.

    This includes connection errors, timeout errors, and generation failures.
    """

    pass


class APIError(ChessRAGException):
    """
    Errors in the FastAPI application.

    This includes request validation errors and response errors.
    """

    pass


class ValidationError(ChessRAGException):
    """
    Data validation errors.

    Raised when input data doesn't meet expected schema or constraints.
    """

    pass
