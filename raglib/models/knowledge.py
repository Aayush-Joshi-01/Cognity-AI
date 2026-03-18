"""Knowledge graph models: Entity, Relation, ExtractionResult."""
from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum


class SourceStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DEPRECATED = "deprecated"


class Entity(BaseModel):
    name: str
    entity_type: str
    description: str = ""
    properties: dict = Field(default_factory=dict)
    source_id: str = ""
    confidence: float = 1.0
    extraction_method: str = "nlp"  # "nlp" | "llm" | "merged"
    mentions: int = 1


class Relation(BaseModel):
    source_entity: str
    relation_type: str
    target_entity: str
    description: str = ""
    properties: dict = Field(default_factory=dict)
    source_id: str = ""
    confidence: float = 1.0
    extraction_method: str = "nlp"
    weight: float = 1.0


class ExtractionResult(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
