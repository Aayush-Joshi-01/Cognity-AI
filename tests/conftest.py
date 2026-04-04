"""Shared fixtures and helpers for the cognity-ai test suite."""
from __future__ import annotations

import os
import tempfile
import random
from pathlib import Path

import pytest

# ── Credential guards ────────────────────────────────────────────────────────

HAS_GEMINI_KEY = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
HAS_ANTHROPIC_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))
HAS_COHERE_KEY = bool(os.getenv("COHERE_API_KEY"))
HAS_PINECONE_KEY = bool(os.getenv("PINECONE_API_KEY"))
HAS_QDRANT_KEY = bool(os.getenv("QDRANT_API_KEY") or os.getenv("QDRANT_URL"))
HAS_AWS_CREDS = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
HAS_AZURE_OPENAI = bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_KEY"))


# ── Availability guards (optional library deps) ──────────────────────────────

def _importable(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


HAS_FAISS = _importable("faiss")
HAS_CHROMADB = _importable("chromadb")
HAS_NETWORKX = _importable("networkx")
HAS_SPACY = _importable("spacy")
HAS_SENTENCE_TRANSFORMERS = _importable("sentence_transformers")
HAS_PINECONE_PKG = _importable("pinecone")
HAS_QDRANT_PKG = _importable("qdrant_client")


# ── Pytest marks ─────────────────────────────────────────────────────────────

requires_gemini = pytest.mark.skipif(
    not HAS_GEMINI_KEY, reason="GOOGLE_API_KEY / GEMINI_API_KEY not set"
)
requires_openai = pytest.mark.skipif(
    not HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set"
)
requires_anthropic = pytest.mark.skipif(
    not HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set"
)
requires_cohere = pytest.mark.skipif(
    not HAS_COHERE_KEY, reason="COHERE_API_KEY not set"
)
requires_pinecone = pytest.mark.skipif(
    not (HAS_PINECONE_KEY and HAS_PINECONE_PKG),
    reason="PINECONE_API_KEY not set or pinecone-client not installed",
)
requires_aws = pytest.mark.skipif(
    not HAS_AWS_CREDS, reason="AWS credentials not set"
)
requires_azure_openai = pytest.mark.skipif(
    not HAS_AZURE_OPENAI, reason="Azure OpenAI credentials not set"
)
requires_faiss = pytest.mark.skipif(not HAS_FAISS, reason="faiss not installed")
requires_chromadb = pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")
requires_networkx = pytest.mark.skipif(not HAS_NETWORKX, reason="networkx not installed")
requires_spacy = pytest.mark.skipif(not HAS_SPACY, reason="spacy not installed")
requires_sentence_transformers = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed"
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    """Return a temporary directory path string."""
    return str(tmp_path)


@pytest.fixture
def sample_text():
    return (
        "Alice works at Acme Corp in New York. "
        "Bob is a scientist at MIT. "
        "Alice and Bob collaborated on a machine learning project. "
        "The project was funded by the National Science Foundation. "
        "Results were published in Nature magazine."
    )


@pytest.fixture
def sample_text_file(tmp_path, sample_text):
    """Write sample text to a .txt file and return its path."""
    p = tmp_path / "sample.txt"
    p.write_text(sample_text, encoding="utf-8")
    return str(p)


@pytest.fixture
def sample_md_file(tmp_path):
    content = "# Introduction\n\nThis is the intro.\n\n## Methods\n\nThese are the methods.\n\n## Results\n\nHere are the results.\n"
    p = tmp_path / "sample.md"
    p.write_text(content, encoding="utf-8")
    return str(p)


def make_embedding(dim: int = 8, seed: int | None = None) -> list[float]:
    """Return a deterministic unit-ish embedding vector for testing."""
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return [x / norm for x in vec]


@pytest.fixture
def tiny_embedding():
    """8-dim normalised embedding for fast store tests."""
    return make_embedding(dim=8, seed=42)
