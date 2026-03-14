from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class DocumentType(StrEnum):
    """Enumeration of supported document types."""

    ARTICLE = "article"
    PGN = "pgn"
    BOOK = "book"
    ANNOTATION = "annotation"


class Document(BaseModel):
    """Domain model representing a stored document.

    Attributes:
        id: Unique identifier for the document.
        content: Raw text content of the document.
        doc_type: Type of document (see `DocumentType`).
        metadata: Arbitrary metadata associated with the document.
        embedding: Optional embedding vector for semantic search.
        created_at: UTC timestamp when the document was created.
    """

    id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    doc_type: DocumentType = Field(...)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("embedding")
    def _check_embedding(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("embedding must be a list of floats")
        for i, item in enumerate(v):
            try:
                float(item)
            except Exception as e:
                raise ValueError(f"embedding element at index {i} is not a float: {e}")
        return v


class ChessQuery(BaseModel):
    """Model describing a user query against the chess knowledge base."""

    query: str = Field(..., min_length=1, max_length=1000)
    context: list[str] = Field(default_factory=list)
    max_results: int = Field(5, ge=1, le=20)
    filters: dict[str, Any] | None = None

    @field_validator("context")
    def _validate_context(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("context must be a list of strings")
        if len(v) > 50:
            raise ValueError("context list is too long (max 50 entries)")
        return v


class QueryResponse(BaseModel):
    """Model for responses returned from query execution.

    Contains the final answer, source attributions, intermediate reasoning
    steps and tooling information useful for debugging/traceability.
    """

    query: str = Field(...)
    answer: str = Field(..., min_length=0)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_steps: list[dict[str, Any]] = Field(default_factory=list)
    tools_used: list[str] = Field(default_factory=list)
    response_time_ms: int = Field(..., ge=0)

    @field_validator("response_time_ms")
    def _validate_response_time(cls, v: int) -> int:
        if v < 0:
            raise ValueError("response_time_ms must be non-negative")
        return int(v)


__all__ = ["DocumentType", "Document", "ChessQuery", "QueryResponse"]
