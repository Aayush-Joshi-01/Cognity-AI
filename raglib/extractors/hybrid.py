"""HybridExtractor — NLP first, LLM fills semantic gaps (DEFAULT)."""
from raglib.extractors.base import BaseExtractor
from raglib.extractors.nlp import NLPExtractor
from raglib.models.knowledge import ExtractionResult


class HybridExtractor(BaseExtractor):
    """Combines NLPExtractor and LLMExtractor for maximum coverage.

    Modes:
    - ``"augment"`` (default): Run NLP first, then LLM fills semantic gaps.
    - ``"nlp_only"``: Skip the LLM step entirely.
    - ``"llm_only"``: Skip NLP and use LLM directly (falls back to NLP if no LLM).
    """

    def __init__(
        self,
        nlp_extractor: NLPExtractor,
        llm_extractor=None,
        mode: str = "augment",
    ):
        """
        Args:
            nlp_extractor: A configured NLPExtractor instance.
            llm_extractor: An optional LLMExtractor instance.
            mode: Extraction mode — "augment" | "nlp_only" | "llm_only".
        """
        self.nlp = nlp_extractor
        self.llm = llm_extractor
        self.mode = mode

    # ── BaseExtractor interface ──────────────────────────────────────────

    def extract(self, text: str, source_id: str = "") -> ExtractionResult:
        """Extract entities and relations using the configured mode.

        Args:
            text: Input text.
            source_id: Document/source identifier attached to extracted items.

        Returns:
            Merged and deduplicated ExtractionResult.
        """
        if self.mode == "nlp_only":
            return self.nlp.extract(text, source_id)

        if self.mode == "llm_only":
            if self.llm is not None:
                return self.llm.extract(text, source_id)
            # Graceful fallback: no LLM configured
            return self.nlp.extract(text, source_id)

        # Default: "augment" — NLP first, LLM fills gaps
        nlp_result = self.nlp.extract(text, source_id)
        if self.llm is not None:
            llm_result = self._llm_augment(text, nlp_result, source_id)
            nlp_result.entities.extend(llm_result.entities)
            nlp_result.relations.extend(llm_result.relations)
            # Deduplicate merged results
            nlp_result.entities = self.nlp._deduplicate_entities(nlp_result.entities)
            nlp_result.relations = self.nlp._deduplicate_relations(nlp_result.relations)
        return nlp_result

    # ── LLM augmentation ─────────────────────────────────────────────────

    def _llm_augment(
        self,
        text: str,
        nlp_result: ExtractionResult,
        source_id: str,
    ) -> ExtractionResult:
        """Ask the LLM to find entities/relations the NLP missed.

        Passes existing NLP findings as context so the LLM only returns
        genuinely new items.
        """
        from raglib.extractors.llm import AUGMENT_EXTRACTION_PROMPT
        import json

        existing_ents = ", ".join(e.name for e in nlp_result.entities[:30])
        existing_rels = "; ".join(
            f"{r.source_entity}-[{r.relation_type}]->{r.target_entity}"
            for r in nlp_result.relations[:20]
        )

        prompt = AUGMENT_EXTRACTION_PROMPT.format(
            existing_entities=existing_ents or "None",
            existing_relations=existing_rels or "None",
            text=text[:3000],
        )

        # Call LLM through its internal _call_generator to reuse the same logic
        raw_text = self.llm._call_generator(prompt)
        raw_text = self.llm._strip_markdown(raw_text)

        try:
            raw = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            return ExtractionResult()

        from raglib.models.knowledge import Entity, Relation

        entities = [
            Entity(
                name=e["name"].strip().title(),
                entity_type=e.get("entity_type", "Other"),
                description=e.get("description", ""),
                source_id=source_id,
                confidence=0.9,
                extraction_method="llm",
            )
            for e in raw.get("entities", [])
            if "name" in e
        ]
        relations = [
            Relation(
                source_entity=r["source_entity"].strip().title(),
                relation_type=r["relation_type"].strip().upper().replace(" ", "_"),
                target_entity=r["target_entity"].strip().title(),
                description=r.get("description", ""),
                source_id=source_id,
                confidence=0.85,
                extraction_method="llm",
            )
            for r in raw.get("relations", [])
            if "source_entity" in r and "relation_type" in r and "target_entity" in r
        ]
        return ExtractionResult(entities=entities, relations=relations)
