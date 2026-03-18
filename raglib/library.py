"""
RAGLibrary — unified public API for the raglib modular RAG service.

Single entry point for:
  - Ingesting any file format (PDF, DOCX, XLSX, PPTX, TXT, MD, images, …)
  - Querying with any RAG methodology
  - Lifecycle management (confirm, deprecate, prune)
  - Plugin registration (custom loaders, chunkers, embedders, etc.)

Example:
    from raglib import RAGLibrary

    rag = RAGLibrary(
        rag_method="hybrid_graph",
        embedder="gemini",
        llm="gemini",
        gemini_api_key="...",
    )
    rag.ingest("report.pdf")
    rag.ingest("data.xlsx")
    answer = rag.query("What are the main findings?")
"""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Type

from raglib.config.base import LibraryConfig
from raglib.config.providers import (
    GeminiConfig, Neo4jConfig, ChromaConfig, NLPConfig, GraphRAGConfig,
    IngestionConfig, OpenAIConfig, AnthropicConfig, AzureOpenAIConfig,
    BedrockConfig, VertexAIConfig, QdrantConfig, PineconeConfig,
    MilvusConfig, WeaviateConfig, PgVectorConfig, AzureSearchConfig,
    OllamaConfig, CohereConfig,
)
from raglib.models.document import Document
from raglib.models.retrieval import RetrievalResult, CommunityInfo


class RAGLibrary:
    """
    The primary interface to the raglib modular RAG library.

    All parameters are optional — sensible defaults are chosen automatically.
    """

    def __init__(
        self,
        # ── Component selection ─────────────────────────────────────────
        rag_method: str = "hybrid_graph",
        chunker: str = "sentence",
        embedder: str = "gemini",
        vector_store: str = "chroma",
        graph_store: str = "neo4j",
        llm: str = "gemini",
        extraction: str = "hybrid",
        ocr: str = "gemini_vision",
        page_index: str = "hybrid",
        # ── Provider credentials (shorthand kwargs) ─────────────────────
        gemini_api_key: str = "",
        openai_api_key: str = "",
        anthropic_api_key: str = "",
        azure_openai_endpoint: str = "",
        azure_openai_key: str = "",
        azure_openai_deployment: str = "gpt-4o",
        azure_openai_api_version: str = "2024-02-01",
        aws_region: str = "us-east-1",
        aws_access_key_id: str = "",
        aws_secret_access_key: str = "",
        cohere_api_key: str = "",
        ollama_base_url: str = "http://localhost:11434",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "",
        neo4j_database: str = "neo4j",
        chroma_persist_dir: str = "./chroma_store",
        # ── Full config override ────────────────────────────────────────
        config: LibraryConfig | None = None,
    ):
        # Build or use provided config
        if config is not None:
            self._cfg = config
        else:
            self._cfg = self._build_config(locals())

        # Build all components (lazy import, auto-fallback)
        from raglib.factory import build_components, build_retriever
        self._components = build_components(self._cfg)

        # Build the ingestion pipeline
        from raglib.pipeline.ingestion import IngestionPipeline
        self._pipeline = IngestionPipeline(
            extractor=self._components["extractor"],
            chunker=self._components["chunker"],
            embedder=self._components["embedder"],
            vector_store=self._components["vector_store"],
            graph_store=self._components["graph_store"],
            page_index=self._components["page_index"],
            hash_store=self._components["hash_store"],
            generator=self._components["generator"],
            config=self._cfg,
        )

        # Build the knowledge updater
        if self._components["graph_store"]:
            from raglib.pipeline.knowledge_updater import KnowledgeUpdater
            self._updater = KnowledgeUpdater(
                graph_store=self._components["graph_store"],
                config=self._cfg,
            )
        else:
            self._updater = None

        # Build the default retriever
        self._retriever = build_retriever(
            self._components["rag_method"],
            self._components,
            self._cfg,
        )

        # Loader factory (built lazily on first use)
        self._loader_factory = None

        print(
            f"  raglib ready: method={self._components['rag_method']}, "
            f"embedder={self._cfg.embedder}, "
            f"vector={self._cfg.vector_store}, "
            f"graph={self._cfg.graph_store}"
        )

    # ══════════════════════════════════════════════════════════════════════
    # INGESTION
    # ══════════════════════════════════════════════════════════════════════

    def ingest(
        self,
        source: str | Path,
        doc_id: str | None = None,
        status: str = "pending",
        **meta,
    ) -> dict:
        """
        Ingest any supported file format.

        source: path to a file (.pdf, .docx, .xlsx, .pptx, .txt, .md,
                .jpg, .png, .csv, .html, .json, .yaml, …)
        doc_id: optional explicit document ID (defaults to filename stem)
        status: "pending" | "confirmed" | "deprecated"
        """
        path = Path(source)
        if not doc_id:
            doc_id = path.stem

        factory = self._get_loader_factory()
        docs = factory.load(str(path), ocr=self._components.get("ocr"))
        results = []
        for i, doc in enumerate(docs):
            # For multi-page/multi-sheet files, each Document gets its own sub-id
            sub_id = doc_id if len(docs) == 1 else f"{doc_id}_{i}"
            doc.doc_id = sub_id
            doc.source_name = meta.get("source_name", path.name)
            result = self._pipeline.ingest_document(doc, status=status)
            results.append(result)

        if len(results) == 1:
            return results[0]
        return {
            "doc_id": doc_id, "status": "ingested",
            "parts": len(results),
            "total_entities": sum(r.get("entities", 0) for r in results),
            "total_chunks": sum(r.get("chunks", 0) for r in results),
        }

    def ingest_dir(
        self,
        directory: str | Path,
        glob: str = "**/*",
        status: str = "pending",
        recursive: bool = True,
        **meta,
    ) -> list[dict]:
        """
        Recursively ingest all supported files in a directory.

        Returns a list of ingest results (one per file).
        """
        directory = Path(directory)
        factory = self._get_loader_factory()
        supported_exts = set(factory.supported_extensions())

        pattern = "**/*" if recursive else "*"
        files = [f for f in directory.glob(pattern) if f.is_file()
                 and f.suffix.lower() in supported_exts]

        results = []
        for f in sorted(files):
            try:
                r = self.ingest(f, status=status, **meta)
                results.append(r)
            except Exception as e:
                print(f"  [WARN] Failed to ingest {f.name}: {e}")
                results.append({"doc_id": f.stem, "status": "error", "error": str(e)})

        return results

    def ingest_text(
        self,
        text: str,
        doc_id: str,
        source_name: str = "",
        status: str = "pending",
    ) -> dict:
        """Ingest raw text directly (backward-compatible with hybrid_rag API)."""
        return self._pipeline.ingest(doc_id=doc_id, text=text,
                                      source_name=source_name, status=status)

    def ingest_batch(self, documents: list[dict]) -> list[dict]:
        """Ingest a list of dicts with keys: doc_id, text, source_name, status."""
        return self._pipeline.ingest_batch(documents)

    def build_communities(self) -> list[CommunityInfo]:
        """
        Run knowledge graph community detection + summarization.
        Call after ingesting all documents.
        Requires a graph store that supports Leiden/Louvain (Neo4j + GDS).
        """
        return self._pipeline.build_communities()

    def remove_document(self, doc_id: str):
        """Remove a document and all its graph/vector data."""
        self._pipeline.remove_document(doc_id)

    def sync(self, current_doc_ids: set[str]) -> list[str]:
        """Remove documents no longer in the provided set. Returns list of removed IDs."""
        return self._pipeline.sync(current_doc_ids)

    # ══════════════════════════════════════════════════════════════════════
    # RETRIEVAL & GENERATION
    # ══════════════════════════════════════════════════════════════════════

    def query(self, question: str, top_k: int = 10, method: str | None = None) -> str:
        """
        Query the knowledge base and return a generated answer.

        method: optional per-query RAG method override.
        """
        retriever = self._get_retriever(method)
        return retriever.query(question, top_k=top_k)

    def query_with_sources(
        self, question: str, top_k: int = 10, method: str | None = None
    ) -> dict:
        """
        Query and return answer + source attribution.

        Returns:
            {
                "answer": str,
                "sources": {"graph": [...], "vector": [...], "community": [...]},
                "retrieval_scores": [...],
                "seed_entities": [...],
            }
        """
        retriever = self._get_retriever(method)
        return retriever.query_with_sources(question, top_k=top_k)

    def retrieve(
        self, query: str, top_k: int = 10, method: str | None = None
    ) -> list[RetrievalResult]:
        """Return raw retrieval results without generation."""
        retriever = self._get_retriever(method)
        return retriever.retrieve(query, top_k=top_k)

    # ══════════════════════════════════════════════════════════════════════
    # KNOWLEDGE LIFECYCLE
    # ══════════════════════════════════════════════════════════════════════

    def confirm(self, doc_id: str):
        """Boost confidence of all triples from this source to 1.0."""
        if self._updater:
            self._updater.confirm_source(doc_id)

    def deprecate(self, doc_id: str):
        """Halve confidence of all triples from this source."""
        if self._updater:
            self._updater.deprecate_source(doc_id)

    def bulk_confirm(self, doc_ids: list[str]):
        if self._updater:
            self._updater.bulk_confirm(doc_ids)

    def bulk_deprecate(self, doc_ids: list[str]):
        if self._updater:
            self._updater.bulk_deprecate(doc_ids)

    def detect_conflicts(self, entity_name: str) -> list[dict]:
        """Find contradictions for a given entity across sources."""
        if self._updater:
            return self._updater.detect_conflicts(entity_name)
        return []

    def prune(self, threshold: float = 0.5) -> int:
        """Remove relations below the confidence threshold. Returns count removed."""
        if self._updater:
            return self._updater.prune_low_confidence(threshold)
        return 0

    def health_report(self) -> dict:
        """Return overall knowledge base health statistics."""
        if self._updater:
            return self._updater.health_report()
        # Vector-only health
        return {"status": "vector-only", "graph_store": None}

    def source_stats(self) -> list[dict]:
        """Per-document statistics (requires Neo4j/Memgraph graph store)."""
        if self._updater:
            return self._updater.get_source_stats()
        return []

    # ══════════════════════════════════════════════════════════════════════
    # PLUGIN REGISTRATION
    # ══════════════════════════════════════════════════════════════════════

    def register_loader(self, ext: str, loader_class: Type):
        """Register a custom loader for a file extension. e.g. '.myext'"""
        from raglib.registry import PluginRegistry
        PluginRegistry.register_loader(ext, loader_class)

    def register_chunker(self, name: str, chunker_class: Type):
        from raglib.registry import PluginRegistry
        PluginRegistry.register_chunker(name, chunker_class)

    def register_embedder(self, name: str, embedder_class: Type):
        from raglib.registry import PluginRegistry
        PluginRegistry.register_embedder(name, embedder_class)

    def register_generator(self, name: str, generator_class: Type):
        from raglib.registry import PluginRegistry
        PluginRegistry.register_generator(name, generator_class)

    def register_retriever(self, name: str, retriever_class: Type):
        from raglib.registry import PluginRegistry
        PluginRegistry.register_retriever(name, retriever_class)

    def available_plugins(self) -> dict:
        """List all registered plugins."""
        from raglib.registry import PluginRegistry
        return PluginRegistry.summary()

    # ══════════════════════════════════════════════════════════════════════
    # COMPONENT ACCESS (for advanced use)
    # ══════════════════════════════════════════════════════════════════════

    @property
    def pipeline(self):
        """Direct access to the IngestionPipeline."""
        return self._pipeline

    @property
    def retriever(self):
        """Direct access to the default retriever."""
        return self._retriever

    @property
    def updater(self):
        """Direct access to the KnowledgeUpdater (None if no graph store)."""
        return self._updater

    @property
    def config(self) -> LibraryConfig:
        """The full library configuration."""
        return self._cfg

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _get_loader_factory(self):
        if self._loader_factory is None:
            from raglib.loaders.factory import LoaderFactory
            self._loader_factory = LoaderFactory()
        return self._loader_factory

    def _get_retriever(self, method: str | None):
        """Return default retriever or build a per-query override."""
        if method is None or method == self._components["rag_method"]:
            return self._retriever
        # Build a temporary retriever for this specific query
        from raglib.factory import build_retriever
        return build_retriever(method, self._components, self._cfg)

    @staticmethod
    def _build_config(kwargs: dict) -> LibraryConfig:
        """Construct LibraryConfig from __init__ keyword arguments."""
        cfg = LibraryConfig(
            rag_method=kwargs.get("rag_method", "hybrid_graph"),
            chunker=kwargs.get("chunker", "sentence"),
            embedder=kwargs.get("embedder", "gemini"),
            vector_store=kwargs.get("vector_store", "chroma"),
            graph_store=kwargs.get("graph_store", "neo4j"),
            llm=kwargs.get("llm", "gemini"),
            extraction=kwargs.get("extraction", "hybrid"),
            ocr=kwargs.get("ocr", "gemini_vision"),
            page_index=kwargs.get("page_index", "hybrid"),
        )

        # Apply credential overrides
        if kwargs.get("gemini_api_key"):
            cfg.gemini.api_key = kwargs["gemini_api_key"]
        if kwargs.get("openai_api_key"):
            cfg.openai.api_key = kwargs["openai_api_key"]
        if kwargs.get("anthropic_api_key"):
            cfg.anthropic.api_key = kwargs["anthropic_api_key"]
        if kwargs.get("azure_openai_endpoint"):
            cfg.azure_openai.endpoint = kwargs["azure_openai_endpoint"]
        if kwargs.get("azure_openai_key"):
            cfg.azure_openai.api_key = kwargs["azure_openai_key"]
        if kwargs.get("azure_openai_deployment"):
            cfg.azure_openai.deployment_name = kwargs["azure_openai_deployment"]
        if kwargs.get("azure_openai_api_version"):
            cfg.azure_openai.api_version = kwargs["azure_openai_api_version"]
        if kwargs.get("aws_region"):
            cfg.bedrock.region = kwargs["aws_region"]
        if kwargs.get("aws_access_key_id"):
            cfg.bedrock.access_key_id = kwargs["aws_access_key_id"]
        if kwargs.get("aws_secret_access_key"):
            cfg.bedrock.secret_access_key = kwargs["aws_secret_access_key"]
        if kwargs.get("cohere_api_key"):
            cfg.cohere.api_key = kwargs["cohere_api_key"]
        if kwargs.get("ollama_base_url"):
            cfg.ollama.base_url = kwargs["ollama_base_url"]
        if kwargs.get("neo4j_uri"):
            cfg.neo4j.uri = kwargs["neo4j_uri"]
        if kwargs.get("neo4j_user"):
            cfg.neo4j.user = kwargs["neo4j_user"]
        if kwargs.get("neo4j_password"):
            cfg.neo4j.password = kwargs["neo4j_password"]
        if kwargs.get("neo4j_database"):
            cfg.neo4j.database = kwargs["neo4j_database"]
        if kwargs.get("chroma_persist_dir"):
            cfg.chroma.persist_directory = kwargs["chroma_persist_dir"]

        return cfg
