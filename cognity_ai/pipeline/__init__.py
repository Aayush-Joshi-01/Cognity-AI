"""Pipeline package: orchestration of the full ingestion and retrieval pipeline."""

from cognity_ai.pipeline.ingestion import IngestionPipeline
from cognity_ai.pipeline.knowledge_updater import KnowledgeUpdater

__all__ = ["IngestionPipeline", "KnowledgeUpdater"]
