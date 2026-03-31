"""Retrieval and RAG pipeline models."""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from cognity_ai.models.knowledge import SourceStatus


class PageInfo(BaseModel):
    """Page/section index for structural awareness."""
    page_num: int
    section: str = ""
    start_char: int = 0
    end_char: int = 0
    heading: str = ""


class SemanticChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    index: int
    page_info: Optional[PageInfo] = None
    embedding: Optional[list[float]] = None
    entity_names: list[str] = Field(default_factory=list)
    sentence_count: int = 0
    token_estimate: int = 0
    # Parent-child chunking support
    parent_chunk_id: Optional[str] = None
    is_parent: bool = False


class CommunityInfo(BaseModel):
    """Microsoft GraphRAG community structure."""
    community_id: str
    level: int
    entity_names: list[str] = Field(default_factory=list)
    summary: str = ""
    title: str = ""
    parent_community: Optional[str] = None
    rank: float = 0.0
    embedding: Optional[list[float]] = None


class DocumentMeta(BaseModel):
    doc_id: str
    source_name: str
    content_hash: str
    status: SourceStatus = SourceStatus.PENDING
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    chunk_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    page_count: int = 0


class RetrievalResult(BaseModel):
    content: str
    score: float
    source: str  # "graph" | "vector" | "community" | "page" | "vector_bridge"
    metadata: dict = Field(default_factory=dict)
