"""
Gemini layer — only invoked for:
  1. Augmenting NLP extractions with semantic relations spaCy can't catch
  2. Generating entity/relation descriptions (GraphRAG requirement)
  3. Community summarization
  4. Embeddings (batched)

Cost optimization: NLP handles ~70% of extraction; Gemini handles the remaining
semantic nuance + all embeddings.
"""

import json
import time
from typing import Optional

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from config import GeminiConfig
from models import Entity, Relation, ExtractionResult, CommunityInfo


# ── Prompts ─────────────────────────────────────────────────────────────

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


ENTITY_DESCRIPTION_PROMPT = """Generate a concise one-line description for this entity based on context.

Entity: {entity_name} (Type: {entity_type})
Context snippets:
{context}

Return ONLY a single sentence description, nothing else."""


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


class GeminiExtractor:
    def __init__(self, config: GeminiConfig):
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(
            config.model,
            generation_config=genai.GenerationConfig(
                temperature=config.extraction_temperature,
                response_mime_type="application/json",
            ),
        )
        self.text_model = genai.GenerativeModel(
            config.model,
            generation_config=genai.GenerationConfig(temperature=0.1),
        )
        self.embed_model_name = config.embedding_model
        self.batch_limit = config.batch_embed_limit
        self.rpm_limit = config.rpm_limit
        self._last_call = 0.0

    def _rate_limit(self):
        """Simple rate limiter — ensures minimum gap between API calls."""
        gap = 60.0 / self.rpm_limit
        elapsed = time.time() - self._last_call
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_call = time.time()

    # ── Augmented Extraction ────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def augment_extraction(
        self, text: str, existing: ExtractionResult, source_id: str = ""
    ) -> ExtractionResult:
        """Find relations/entities that NLP missed."""
        self._rate_limit()

        existing_ents = ", ".join(e.name for e in existing.entities[:30])
        existing_rels = "; ".join(
            f"{r.source_entity}-[{r.relation_type}]->{r.target_entity}"
            for r in existing.relations[:20]
        )

        prompt = AUGMENT_EXTRACTION_PROMPT.format(
            existing_entities=existing_ents or "None",
            existing_relations=existing_rels or "None",
            text=text[:3000],  # cap to control cost
        )
        resp = self.model.generate_content(prompt)
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

    # ── Entity Descriptions (GraphRAG) ──────────────────────────────────

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def generate_entity_description(
        self, entity_name: str, entity_type: str, context_snippets: list[str]
    ) -> str:
        self._rate_limit()
        ctx = "\n".join(f"- {s[:200]}" for s in context_snippets[:5])
        prompt = ENTITY_DESCRIPTION_PROMPT.format(
            entity_name=entity_name, entity_type=entity_type, context=ctx,
        )
        resp = self.text_model.generate_content(prompt)
        return resp.text.strip()

    # ── Community Summarization (GraphRAG) ──────────────────────────────

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=8))
    def summarize_community(
        self, entity_names: list[str], relation_descriptions: list[str]
    ) -> dict:
        self._rate_limit()
        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            entities=", ".join(entity_names),
            relations="\n".join(f"- {r}" for r in relation_descriptions[:20]),
        )
        resp = self.model.generate_content(prompt)
        return json.loads(resp.text)

    # ── Embeddings (batched for cost) ───────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def embed_batch(self, texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
        """Batch embed with automatic chunking to respect API limits."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_limit):
            batch = texts[i:i + self.batch_limit]
            self._rate_limit()
            result = genai.embed_content(
                model=self.embed_model_name,
                content=batch,
                task_type=task_type,
            )
            all_embeddings.extend(result["embedding"])
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        result = genai.embed_content(
            model=self.embed_model_name,
            content=query,
            task_type="retrieval_query",
        )
        return result["embedding"]
