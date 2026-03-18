"""GeminiGenerator — default generator using Google Gemini models.

Also exposes augment_extraction() and summarize_community() so that the
same Gemini client can serve both generation and knowledge-graph extraction
tasks without requiring a separate GeminiExtractor instance.
"""
import json
import time

from raglib.generators.base import BaseGenerator, GENERATION_PROMPT


# ── Extraction prompts (copied from hybrid_rag/gemini_extractor.py) ─────────

AUGMENT_EXTRACTION_PROMPT = """You are given text and pre-extracted entities/relations from NLP.
Your job: find ADDITIONAL semantic relationships the NLP missed. Do NOT repeat what's already extracted.

Already extracted entities: {existing_entities}
Already extracted relations: {existing_relations}

Text:
{text}

Return ONLY valid JSON (no markdown):
{{
  "entities": [
    {{"name": "Name", "entity_type": "Person|Organization|Technology|Concept|Location|Event|Other", "description": "one-line description"}}
  ],
  "relations": [
    {{"source_entity": "A", "relation_type": "UPPER_SNAKE_CASE", "target_entity": "B", "description": "how they relate"}}
  ]
}}

Focus on: causal links, temporal sequences, part-of hierarchies, and implicit associations.
Only return NEW findings. Return empty lists if nothing new."""


COMMUNITY_SUMMARY_PROMPT = """Summarize this community of related entities and their relationships.
This summary will be used for high-level retrieval, so capture the key themes and connections.

Community entities: {entities}
Key relationships:
{relations}

Return a JSON object (no markdown):
{{
  "title": "2-5 word title for this community",
  "summary": "2-3 sentence summary of what this community represents and its key dynamics"
}}"""


class GeminiGenerator(BaseGenerator):
    """Generate answers using Google Gemini via the google-generativeai SDK.

    Two model handles are maintained:
      gen_model  — standard text generation (temperature-controlled)
      json_model — JSON-mode model used for structured extraction tasks
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        extraction_temperature: float = 0.0,
        rpm_limit: int = 15,
    ):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.gen_model = genai.GenerativeModel(
            model,
            generation_config=genai.GenerationConfig(temperature=temperature),
        )
        self.json_model = genai.GenerativeModel(
            model,
            generation_config=genai.GenerationConfig(
                temperature=extraction_temperature,
                response_mime_type="application/json",
            ),
        )
        self._rpm_limit = rpm_limit
        self._last_call = 0.0

    def _rate_limit(self):
        """Ensure minimum gap between API calls to honour rpm_limit."""
        gap = 60.0 / self._rpm_limit
        elapsed = time.time() - self._last_call
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_call = time.time()

    # ── BaseGenerator interface ──────────────────────────────────────────

    def generate(self, question: str, context: str) -> str:
        """Generate an answer for question using the provided context string."""
        self._rate_limit()
        if question:
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            # generate_with_structured_context passes a pre-built prompt as context
            prompt = context
        resp = self.gen_model.generate_content(prompt)
        return resp.text

    def generate_rag(
        self,
        question: str,
        graph_ctx: str = "",
        community_ctx: str = "",
        vector_ctx: str = "",
    ) -> str:
        """Generate using the full three-channel RAG prompt."""
        self._rate_limit()
        prompt = self.build_rag_prompt(question, graph_ctx, community_ctx, vector_ctx)
        resp = self.gen_model.generate_content(prompt)
        return resp.text

    # ── Knowledge-graph helpers ──────────────────────────────────────────

    def augment_extraction(self, text: str, existing, source_id: str = ""):
        """Find entities/relations that NLP missed.

        Parameters
        ----------
        text:
            Source text to analyse (capped internally to 3000 chars).
        existing:
            An ExtractionResult (or any object with .entities/.relations lists)
            containing already-extracted items that should not be repeated.
        source_id:
            Optional document identifier attached to returned objects.

        Returns
        -------
        ExtractionResult
            New entities and relations only.
        """
        from raglib.models.knowledge import Entity, Relation, ExtractionResult

        self._rate_limit()

        existing_ents = ", ".join(e.name for e in existing.entities[:30])
        existing_rels = "; ".join(
            f"{r.source_entity}-[{r.relation_type}]->{r.target_entity}"
            for r in existing.relations[:20]
        )

        prompt = AUGMENT_EXTRACTION_PROMPT.format(
            existing_entities=existing_ents or "None",
            existing_relations=existing_rels or "None",
            text=text[:3000],
        )
        resp = self.json_model.generate_content(prompt)
        raw = json.loads(resp.text)

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
        ]
        return ExtractionResult(entities=entities, relations=relations)

    def summarize_community(
        self, entity_names: list[str], relation_descriptions: list[str]
    ) -> dict:
        """Return a title/summary dict for a graph community.

        Parameters
        ----------
        entity_names:
            Names of entities that form the community.
        relation_descriptions:
            Human-readable descriptions of the key relations (up to 20 used).

        Returns
        -------
        dict
            {"title": str, "summary": str}
        """
        self._rate_limit()
        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            entities=", ".join(entity_names),
            relations="\n".join(f"- {r}" for r in relation_descriptions[:20]),
        )
        resp = self.json_model.generate_content(prompt)
        return json.loads(resp.text)
