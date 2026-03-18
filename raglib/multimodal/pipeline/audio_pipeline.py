"""
Audio ingestion pipeline for multimodal RAG.

Flow:
  Audio file → AudioLoader (segments)
             → Transcription (Whisper local/API, Google STT, AWS Transcribe)
             → Text embedding of transcript via raglib text embedder
             → [Optional] Audio embedding via ImageBind
             → Store in multimodal vector store

.. note::
    This pipeline is **experimental** (beta). APIs may change between versions.

Example::

    from raglib.multimodal.pipeline.audio_pipeline import AudioIngestionPipeline
    from raglib.multimodal.transcribers.whisper_local import WhisperLocalTranscriber
    from raglib.multimodal.stores.chroma_mm import ChromaMultimodalStore
    from raglib.embedders.gemini import GeminiEmbedder

    transcriber = WhisperLocalTranscriber(model_size="base")
    store = ChromaMultimodalStore()
    text_embedder = GeminiEmbedder(api_key="...")

    pipeline = AudioIngestionPipeline(
        transcriber=transcriber,
        store=store,
        text_embedder=text_embedder,
    )
    chunk = pipeline.ingest("interview.mp3")
    print(chunk.full_transcript[:200])
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from raglib.multimodal.models.media import AudioChunk, AudioSegment
from raglib.multimodal.transcribers.base import BaseTranscriber, TranscriptionResult
from raglib.multimodal.stores.base import BaseMultimodalStore

if TYPE_CHECKING:
    from raglib.embedders.base import BaseEmbedder
    from raglib.multimodal.embedders.base import BaseMultimodalEmbedder
    from raglib.multimodal.loaders.audio import AudioLoader

logger = logging.getLogger(__name__)


class AudioIngestionPipeline:
    """
    Orchestrates audio ingestion: load → transcribe → embed → store.

    Args:
        transcriber: A :class:`~raglib.multimodal.transcribers.base.BaseTranscriber`
            instance for speech-to-text.
        store: A :class:`~raglib.multimodal.stores.base.BaseMultimodalStore` for
            persisting audio chunks.
        text_embedder: Optional text embedder (``raglib.embedders.BaseEmbedder``) for
            embedding transcript text.  Enables semantic search over transcripts.
        audio_embedder: Optional multimodal embedder with ``embed_audio()`` support
            (e.g. ``ImageBindEmbedder``).  When provided, audio-level embeddings are
            stored alongside transcript embeddings.
        loader: Optional :class:`~raglib.multimodal.loaders.audio.AudioLoader` instance.
            A default loader is created if *None*.
    """

    def __init__(
        self,
        transcriber: BaseTranscriber,
        store: BaseMultimodalStore,
        text_embedder: "BaseEmbedder | None" = None,
        audio_embedder: "BaseMultimodalEmbedder | None" = None,
        loader: "AudioLoader | None" = None,
    ) -> None:
        self.transcriber = transcriber
        self.store = store
        self.text_embedder = text_embedder
        self.audio_embedder = audio_embedder
        self._loader = loader

    # ------------------------------------------------------------------
    # Lazy-initialise the loader (avoids hard dep at import time)
    # ------------------------------------------------------------------

    @property
    def loader(self) -> "AudioLoader":
        if self._loader is None:
            from raglib.multimodal.loaders.audio import AudioLoader  # lazy import

            self._loader = AudioLoader()
        return self._loader

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        audio_path: str,
        doc_id: str | None = None,
        metadata: dict | None = None,
    ) -> AudioChunk:
        """
        Ingest a single audio file.

        Steps:
        1. Load audio → raw :class:`~raglib.multimodal.models.media.AudioChunk`.
        2. Transcribe the full audio file.
        3. Align transcript segments to audio segments by timestamp.
        4. Embed each segment's transcript text (if ``text_embedder`` provided).
        5. Embed each segment's audio bytes (if ``audio_embedder`` supports audio).
        6. Compute the chunk-level embedding as mean pooling of segment embeddings.
        7. Upsert the completed chunk into ``store``.

        Args:
            audio_path: Absolute or relative path to the audio file.
            doc_id: Optional document identifier.  Defaults to SHA-256 of the file
                content (first 64 KB).
            metadata: Arbitrary key-value metadata attached to the stored chunk.

        Returns:
            The ingested :class:`~raglib.multimodal.models.media.AudioChunk`.

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
            ValueError: If the file extension is not supported by the loader.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        metadata = dict(metadata or {})
        metadata["source_path"] = str(path.resolve())

        # ── Step 1: load ──────────────────────────────────────────────
        logger.info("Loading audio: %s", path)
        audio_chunk = self.loader.load(str(path))

        # Assign doc_id
        if doc_id is None:
            doc_id = self._compute_doc_id(path)
        audio_chunk.id = doc_id
        audio_chunk.audio_id = doc_id
        audio_chunk.source_path = str(path.resolve())
        audio_chunk.metadata.update(metadata)

        # ── Step 2: transcribe ────────────────────────────────────────
        logger.info("Transcribing audio: %s", path)
        transcription: TranscriptionResult = self.transcriber.transcribe(str(path))
        audio_chunk.full_transcript = transcription.full_text
        audio_chunk.metadata["language"] = transcription.language
        audio_chunk.metadata["duration_ms"] = transcription.duration_ms

        # ── Step 3: align transcript segments → audio segments ────────
        self._align_transcription(audio_chunk, transcription)

        # ── Step 4 & 5: embed segments ────────────────────────────────
        segment_embeddings: list[list[float]] = []
        for seg in audio_chunk.segments:
            emb = self._embed_segment(seg)
            if emb:
                seg.embedding = emb
                segment_embeddings.append(emb)

        # ── Step 6: chunk-level mean-pool embedding ───────────────────
        if segment_embeddings:
            try:
                import numpy as np

                arr = np.array(segment_embeddings, dtype=float)
                mean_emb = arr.mean(axis=0)
                norm = float(np.linalg.norm(mean_emb))
                if norm > 0:
                    mean_emb = mean_emb / norm
                audio_chunk.embedding = mean_emb.tolist()
            except ImportError:
                # numpy not available — use first segment embedding
                audio_chunk.embedding = segment_embeddings[0]

        # ── Step 7: persist ───────────────────────────────────────────
        logger.info(
            "Storing audio chunk (id=%s, segments=%d)",
            audio_chunk.id,
            len(audio_chunk.segments),
        )
        self.store.upsert_audio(audio_chunk)

        return audio_chunk

    def ingest_batch(
        self,
        audio_paths: list[str],
        doc_ids: list[str] | None = None,
        metadata: dict | None = None,
    ) -> list[AudioChunk]:
        """
        Ingest multiple audio files sequentially.

        Args:
            audio_paths: List of paths to audio files.
            doc_ids: Optional list of document IDs matching *audio_paths*.
            metadata: Shared metadata applied to all ingested chunks.

        Returns:
            List of ingested :class:`~raglib.multimodal.models.media.AudioChunk` objects.
        """
        doc_ids_iter = doc_ids or [None] * len(audio_paths)
        results: list[AudioChunk] = []
        for path, did in zip(audio_paths, doc_ids_iter):
            try:
                chunk = self.ingest(path, doc_id=did, metadata=metadata)
                results.append(chunk)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to ingest audio %s: %s", path, exc)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_doc_id(self, path: Path) -> str:
        """Return SHA-256 hex of the file's first 64 KB."""
        h = hashlib.sha256()
        with path.open("rb") as fh:
            h.update(fh.read(65_536))
        return h.hexdigest()

    def _align_transcription(
        self,
        audio_chunk: AudioChunk,
        transcription: TranscriptionResult,
    ) -> None:
        """
        Fill each ``AudioSegment.transcript`` by matching its time window to
        ``TimestampedSegment`` entries.

        Falls back to assigning the full transcript to the first segment when no
        per-segment timestamps are available.
        """
        if not transcription.segments:
            if audio_chunk.segments:
                audio_chunk.segments[0].transcript = transcription.full_text
            return

        for audio_seg in audio_chunk.segments:
            matched: list[str] = []
            for ts_seg in transcription.segments:
                # Include if the transcript segment overlaps with this audio window
                if ts_seg.end_ms >= audio_seg.start_ms and ts_seg.start_ms <= audio_seg.end_ms:
                    matched.append(ts_seg.text.strip())
            audio_seg.transcript = " ".join(matched)

    def _embed_segment(self, segment: AudioSegment) -> list[float] | None:
        """
        Compute an embedding for a single audio segment.

        Priority order:
        1. ``audio_embedder.embed_audio()`` on raw audio bytes (e.g. ImageBind).
        2. ``text_embedder.embed_query()`` on the segment transcript.
        3. *None* if neither produces a result.
        """
        # 1. Audio-level embedding (ImageBind style)
        if self.audio_embedder is not None and "audio" in getattr(
            self.audio_embedder, "supported_modalities", []
        ):
            if segment.audio_bytes_b64:
                try:
                    import base64

                    audio_bytes = base64.b64decode(segment.audio_bytes_b64)
                    return self.audio_embedder.embed_audio(audio_bytes)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Audio embedding failed for segment %s: %s", segment.id, exc
                    )

        # 2. Text embedding of transcript (fallback / complementary)
        if self.text_embedder is not None and segment.transcript:
            try:
                return self.text_embedder.embed_query(segment.transcript)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Text embedding failed for segment %s: %s", segment.id, exc
                )

        return None
