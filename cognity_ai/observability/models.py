"""Pydantic event models for the observability system."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Union

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # "native" | "tiktoken" | "estimate"
    source: str = "native"

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            source=self.source if self.source == other.source else "mixed",
        )


class GenerationEvent(BaseModel):
    event_type: str = "generation"
    provider: str
    model: str
    question: str
    answer_length: int
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)


class RetrievalEvent(BaseModel):
    event_type: str = "retrieval"
    rag_method: str
    query: str
    top_k: int
    results_count: int
    channels_used: list[str] = Field(default_factory=list)
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EmbedEvent(BaseModel):
    event_type: str = "embed"
    provider: str
    model: str
    batch_size: int
    token_estimate: int
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Union type for type-safe fan-out
ObservabilityEvent = Annotated[
    Union[GenerationEvent, RetrievalEvent, EmbedEvent],
    Field(discriminator="event_type"),
]
