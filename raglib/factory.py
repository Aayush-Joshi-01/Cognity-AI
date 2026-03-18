"""
ComponentFactory — wires all raglib components from LibraryConfig.

Implements smart defaults, lazy imports, and auto-fallback:
  - hybrid_graph → naive if no graph store available
  - anthropic embedder → sentence_transformers (Anthropic has no embedding API)
  - faiss vector store → community search disabled
  - networkx graph store → community detection disabled
"""
from __future__ import annotations
import warnings
from raglib.config.base import LibraryConfig


def build_components(cfg: LibraryConfig) -> dict:
    """
    Build and return all pipeline components as a dict:
      - nlp_model:      spaCy Language (if NLP-based extraction or sentence chunker)
      - extractor:      BaseExtractor
      - chunker:        BaseChunker
      - embedder:       BaseEmbedder
      - vector_store:   BaseVectorStore
      - graph_store:    BaseGraphStore | None
      - generator:      BaseGenerator
      - page_index:     BasePageIndex
      - hash_store:     HashStore
      - ocr:            BaseOCR
    """
    components = {}

    # ── spaCy NLP model (shared by NLP-based extractor + sentence chunker) ──
    nlp_model = _build_nlp_model(cfg)
    components["nlp_model"] = nlp_model

    # ── OCR ────────────────────────────────────────────────────────────────
    components["ocr"] = _build_ocr(cfg)

    # ── Embedder ───────────────────────────────────────────────────────────
    # Note: anthropic has no embedding API → auto-switch
    embedder_key = cfg.embedder
    if embedder_key == "anthropic":
        warnings.warn(
            "Anthropic has no embedding API. Switching embedder to 'sentence_transformers'.",
            UserWarning, stacklevel=3,
        )
        embedder_key = "sentence_transformers"
    components["embedder"] = _build_embedder(embedder_key, cfg)

    # ── Generator (LLM) ────────────────────────────────────────────────────
    components["generator"] = _build_generator(cfg.llm, cfg)

    # ── Graph Store ────────────────────────────────────────────────────────
    graph_store = _build_graph_store(cfg)
    components["graph_store"] = graph_store

    # ── Vector Store ───────────────────────────────────────────────────────
    components["vector_store"] = _build_vector_store(cfg)

    # ── Extractor ──────────────────────────────────────────────────────────
    components["extractor"] = _build_extractor(cfg, nlp_model, components["generator"])

    # ── Page Index ─────────────────────────────────────────────────────────
    components["page_index"] = _build_page_index(cfg)

    # ── Chunker ────────────────────────────────────────────────────────────
    components["chunker"] = _build_chunker(cfg, nlp_model, components["embedder"])

    # ── Hash Store ─────────────────────────────────────────────────────────
    from raglib.utils.hash import HashStore
    components["hash_store"] = HashStore(path=cfg.ingestion.hash_store_path)

    # ── RAG method selection / fallback ────────────────────────────────────
    rag_method = cfg.rag_method
    if rag_method == "hybrid_graph" and graph_store is None:
        warnings.warn(
            "rag_method='hybrid_graph' requires a graph store, but none is available. "
            "Falling back to 'naive'.",
            UserWarning, stacklevel=3,
        )
        rag_method = "naive"
    components["rag_method"] = rag_method

    return components


# ══════════════════════════════════════════════════════════════════════════
# BUILDER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def _build_nlp_model(cfg: LibraryConfig):
    """Load spaCy model; fall back to smaller model if needed."""
    if cfg.extraction == "llm_only":
        return None  # no spaCy needed for pure LLM extraction
    try:
        import spacy
        try:
            return spacy.load(cfg.nlp.spacy_model)
        except OSError:
            try:
                return spacy.load(cfg.nlp.fallback_model)
            except OSError:
                import subprocess, sys
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", cfg.nlp.fallback_model],
                    check=False,
                )
                return spacy.load(cfg.nlp.fallback_model)
    except ImportError:
        warnings.warn("spaCy not installed. NLP extraction disabled; using LLM-only.", UserWarning)
        return None


def _build_ocr(cfg: LibraryConfig):
    from raglib.ocr.factory import OCRFactory
    try:
        return OCRFactory.create(cfg.ocr, cfg)
    except Exception as e:
        warnings.warn(f"OCR provider '{cfg.ocr}' failed to initialize: {e}. "
                      "OCR will fall back to tesseract or be unavailable.", UserWarning)
        try:
            return OCRFactory.create("tesseract", cfg)
        except Exception:
            return None


def _build_embedder(key: str, cfg: LibraryConfig):
    if key == "gemini":
        from raglib.embedders.gemini import GeminiEmbedder
        return GeminiEmbedder(cfg.gemini)
    elif key == "vertex_ai":
        from raglib.embedders.vertex_ai import VertexAIEmbedder
        return VertexAIEmbedder(cfg.vertex_ai)
    elif key == "openai":
        from raglib.embedders.openai import OpenAIEmbedder
        return OpenAIEmbedder(cfg.openai)
    elif key == "azure_openai":
        from raglib.embedders.azure_openai import AzureOpenAIEmbedder
        return AzureOpenAIEmbedder(cfg.azure_openai)
    elif key == "bedrock":
        from raglib.embedders.bedrock import BedrockEmbedder
        return BedrockEmbedder(cfg.bedrock)
    elif key == "cohere":
        from raglib.embedders.cohere import CohereEmbedder
        return CohereEmbedder(cfg.cohere)
    elif key == "sentence_transformers":
        from raglib.embedders.sentence_transformers import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder()
    elif key == "ollama":
        from raglib.embedders.ollama import OllamaEmbedder
        return OllamaEmbedder(cfg.ollama)
    else:
        raise ValueError(f"Unknown embedder: '{key}'. "
                         f"Choose from: gemini, vertex_ai, openai, azure_openai, bedrock, "
                         f"cohere, sentence_transformers, ollama")


def _build_generator(key: str, cfg: LibraryConfig):
    if key == "gemini":
        from raglib.generators.gemini import GeminiGenerator
        return GeminiGenerator(cfg.gemini)
    elif key == "vertex_ai":
        from raglib.generators.vertex_ai import VertexAIGenerator
        return VertexAIGenerator(cfg.vertex_ai)
    elif key == "openai":
        from raglib.generators.openai import OpenAIGenerator
        return OpenAIGenerator(cfg.openai)
    elif key == "azure_openai":
        from raglib.generators.azure_openai import AzureOpenAIGenerator
        return AzureOpenAIGenerator(cfg.azure_openai)
    elif key == "anthropic":
        from raglib.generators.anthropic import AnthropicGenerator
        return AnthropicGenerator(cfg.anthropic)
    elif key == "bedrock":
        from raglib.generators.bedrock import BedrockGenerator
        return BedrockGenerator(cfg.bedrock)
    elif key == "cohere":
        from raglib.generators.cohere import CohereGenerator
        return CohereGenerator(cfg.cohere)
    elif key == "ollama":
        from raglib.generators.ollama import OllamaGenerator
        return OllamaGenerator(cfg.ollama)
    else:
        raise ValueError(f"Unknown LLM generator: '{key}'. "
                         f"Choose from: gemini, vertex_ai, openai, azure_openai, anthropic, "
                         f"bedrock, cohere, ollama")


def _build_graph_store(cfg: LibraryConfig):
    key = cfg.graph_store
    if not key or key == "none":
        return None
    try:
        if key == "neo4j":
            from raglib.stores.graph.neo4j import Neo4jStore
            return Neo4jStore(cfg.neo4j, cfg.graphrag)
        elif key == "microsoft_graphrag":
            from raglib.stores.graph.microsoft_graphrag import MicrosoftGraphRAGStore
            working_dir = getattr(cfg, "ms_graphrag_dir", "./graphrag_workspace")
            return MicrosoftGraphRAGStore(working_dir=working_dir)
        elif key == "memgraph":
            from raglib.stores.graph.memgraph import MemgraphStore
            return MemgraphStore(cfg.neo4j, cfg.graphrag)  # reuses neo4j config fields
        elif key == "arangodb":
            from raglib.stores.graph.arangodb import ArangoDBStore
            arango_cfg = getattr(cfg, "arangodb", None) or cfg.neo4j
            return ArangoDBStore(arango_cfg)
        elif key == "networkx":
            from raglib.stores.graph.networkx import NetworkXStore
            return NetworkXStore()
        else:
            raise ValueError(f"Unknown graph store: '{key}'. "
                             f"Choose from: neo4j, microsoft_graphrag, memgraph, arangodb, networkx, none")
    except Exception as e:
        warnings.warn(
            f"Graph store '{key}' failed to initialize: {e}. "
            "Graph features disabled — falling back to vector-only.", UserWarning,
        )
        return None


def _build_vector_store(cfg: LibraryConfig):
    key = cfg.vector_store
    if key == "chroma":
        from raglib.stores.vector.chroma import ChromaStore
        return ChromaStore(cfg.chroma)
    elif key == "qdrant":
        from raglib.stores.vector.qdrant import QdrantStore
        return QdrantStore(cfg.qdrant)
    elif key == "pinecone":
        from raglib.stores.vector.pinecone import PineconeStore
        return PineconeStore(cfg.pinecone)
    elif key == "faiss":
        from raglib.stores.vector.faiss import FAISSStore
        return FAISSStore(cfg.qdrant)  # uses dimension from qdrant config
    elif key == "weaviate":
        from raglib.stores.vector.weaviate import WeaviateStore
        return WeaviateStore(cfg.weaviate)
    elif key == "milvus":
        from raglib.stores.vector.milvus import MilvusStore
        return MilvusStore(cfg.milvus)
    elif key == "pgvector":
        from raglib.stores.vector.pgvector import PgVectorStore
        return PgVectorStore(cfg.pgvector)
    elif key == "azure_search":
        from raglib.stores.vector.azure_search import AzureSearchStore
        return AzureSearchStore(cfg.azure_search)
    else:
        raise ValueError(f"Unknown vector store: '{key}'. "
                         f"Choose from: chroma, qdrant, pinecone, faiss, weaviate, milvus, "
                         f"pgvector, azure_search")


def _build_extractor(cfg: LibraryConfig, nlp_model, generator):
    mode = cfg.extraction
    if mode == "nlp_only":
        if nlp_model is None:
            warnings.warn("nlp_only extraction requested but spaCy unavailable. "
                          "Falling back to llm_only.", UserWarning)
            from raglib.extractors.llm import LLMExtractor
            return LLMExtractor(generator)
        from raglib.extractors.nlp import NLPExtractor
        return NLPExtractor(nlp_model, cfg.nlp)
    elif mode == "llm_only":
        from raglib.extractors.llm import LLMExtractor
        return LLMExtractor(generator)
    else:  # "hybrid" (default)
        from raglib.extractors.hybrid import HybridExtractor
        from raglib.extractors.nlp import NLPExtractor
        from raglib.extractors.llm import LLMExtractor
        nlp_ext = NLPExtractor(nlp_model, cfg.nlp) if nlp_model else None
        llm_ext = LLMExtractor(generator)
        if nlp_ext is None:
            return llm_ext
        return HybridExtractor(
            nlp_extractor=nlp_ext,
            llm_extractor=llm_ext,
            mode="augment" if cfg.ingestion.gemini_extraction_mode == "augment" else "hybrid",
        )


def _build_page_index(cfg: LibraryConfig):
    key = cfg.page_index
    if key == "regex":
        from raglib.page_index.regex_index import RegexPageIndex
        return RegexPageIndex(store_path=cfg.ingestion.page_index_path)
    elif key == "structural":
        from raglib.page_index.structural_index import StructuralPageIndex
        return StructuralPageIndex(store_path=cfg.ingestion.page_index_path)
    else:  # "hybrid" (default)
        from raglib.page_index.hybrid_index import HybridPageIndex
        return HybridPageIndex(store_path=cfg.ingestion.page_index_path)


def _build_chunker(cfg: LibraryConfig, nlp_model, embedder):
    key = cfg.chunker
    chunk_cfg = cfg.nlp
    if key == "sentence":
        from raglib.chunkers.sentence import SentenceChunker
        return SentenceChunker(
            nlp_model=nlp_model,
            chunk_sentences=chunk_cfg.semantic_chunk_sentences,
            overlap=chunk_cfg.semantic_chunk_overlap,
        )
    elif key == "fixed":
        from raglib.chunkers.fixed import FixedChunker
        return FixedChunker()
    elif key == "recursive":
        from raglib.chunkers.recursive import RecursiveChunker
        return RecursiveChunker()
    elif key == "semantic":
        from raglib.chunkers.semantic import SemanticChunker
        return SemanticChunker(embedder=embedder, nlp_model=nlp_model)
    elif key == "parent_child":
        from raglib.chunkers.parent_child import ParentChildChunker
        return ParentChildChunker()
    elif key == "hybrid":
        from raglib.chunkers.hybrid import HybridChunker
        return HybridChunker(nlp_model=nlp_model, embedder=embedder)
    else:
        raise ValueError(f"Unknown chunker: '{key}'. "
                         f"Choose from: sentence, fixed, recursive, semantic, parent_child, hybrid")


def build_retriever(rag_method: str, components: dict, cfg: LibraryConfig):
    """Build the retriever for the given RAG method."""
    embedder = components["embedder"]
    vector_store = components["vector_store"]
    graph_store = components["graph_store"]
    generator = components["generator"]
    nlp_extractor = components["extractor"]

    if rag_method == "hybrid_graph":
        from raglib.retrievers.hybrid_graph import HybridGraphRetriever
        return HybridGraphRetriever(
            nlp_extractor=nlp_extractor,
            embedder=embedder,
            vector_store=vector_store,
            graph_store=graph_store,
            generator=generator,
            config=cfg,
        )
    elif rag_method == "naive":
        from raglib.retrievers.naive import NaiveRetriever
        return NaiveRetriever(embedder=embedder, vector_store=vector_store, generator=generator)
    elif rag_method == "vector_only":
        from raglib.retrievers.vector_only import VectorOnlyRetriever
        return VectorOnlyRetriever(embedder=embedder, vector_store=vector_store, generator=generator)
    elif rag_method == "graph_only":
        from raglib.retrievers.graph_only import GraphOnlyRetriever
        return GraphOnlyRetriever(nlp_extractor=nlp_extractor, graph_store=graph_store,
                                   generator=generator)
    elif rag_method == "parent_child":
        from raglib.retrievers.parent_child import ParentChildRetriever
        return ParentChildRetriever(embedder=embedder, vector_store=vector_store,
                                     generator=generator)
    elif rag_method == "multi_query":
        from raglib.retrievers.multi_query import MultiQueryRetriever
        return MultiQueryRetriever(embedder=embedder, vector_store=vector_store,
                                    generator=generator, graph_store=graph_store)
    elif rag_method == "microsoft_graphrag":
        from raglib.retrievers.microsoft_graphrag import MicrosoftGraphRAGRetriever
        working_dir = getattr(cfg, "ms_graphrag_dir", "./graphrag_workspace")
        return MicrosoftGraphRAGRetriever(working_dir=working_dir,
                                           embedder=embedder, generator=generator)
    elif rag_method == "adaptive":
        from raglib.retrievers.adaptive import AdaptiveRetriever
        sub_retrievers = {}
        # Build hybrid_graph if graph available
        if graph_store:
            from raglib.retrievers.hybrid_graph import HybridGraphRetriever
            sub_retrievers["hybrid_graph"] = HybridGraphRetriever(
                nlp_extractor=nlp_extractor, embedder=embedder,
                vector_store=vector_store, graph_store=graph_store,
                generator=generator, config=cfg,
            )
        from raglib.retrievers.naive import NaiveRetriever
        sub_retrievers["naive"] = NaiveRetriever(embedder=embedder, vector_store=vector_store,
                                                   generator=generator)
        default = "hybrid_graph" if graph_store else "naive"
        return AdaptiveRetriever(retrievers=sub_retrievers, default=default)
    else:
        raise ValueError(f"Unknown RAG method: '{rag_method}'. "
                         f"Choose from: hybrid_graph, naive, vector_only, graph_only, "
                         f"parent_child, multi_query, microsoft_graphrag, adaptive")
