"""Pipeline package: orchestration of the full ingestion and retrieval pipeline."""

from raglib.pipeline.ingestion import IngestionPipeline
from raglib.pipeline.knowledge_updater import KnowledgeUpdater

__all__ = ["IngestionPipeline", "KnowledgeUpdater"]
