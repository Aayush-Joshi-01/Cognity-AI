"""LLMExtractor — entity/relation extraction via any generator using AUGMENT_EXTRACTION_PROMPT."""
import json

from raglib.extractors.base import BaseExtractor
from raglib.models.knowledge import Entity, Relation, ExtractionResult


# Identical to gemini_extractor.py AUGMENT_EXTRACTION_PROMPT
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


class LLMExtractor(BaseExtractor):
    """Uses a generator to extract entities and relations via the augmentation prompt.

    The generator must implement one of:
    - ``generate(question: str, context: str) -> str``  (raglib generator interface)
    - ``generate_content(prompt: str)`` with a ``.text`` attribute  (Gemini style)

    For best results the generator should return raw JSON (no markdown fences).
    The extractor attempts to strip markdown code blocks if present.
    """

    def __init__(self, generator):
        """
        Args:
            generator: Any generator object that can produce text from a prompt.
        """
        self._gen = generator

    def extract(self, text: str, source_id: str = "") -> ExtractionResult:
        """Extract entities and relations from text using the LLM.

        Args:
            text: Input text (capped at 3000 chars to control cost).
            source_id: Identifier attached to extracted items.

        Returns:
            ExtractionResult populated from LLM JSON output.
        """
        prompt = AUGMENT_EXTRACTION_PROMPT.format(
            existing_entities="None",
            existing_relations="None",
            text=text[:3000],
        )

        raw_text = self._call_generator(prompt)
        raw_text = self._strip_markdown(raw_text)

        try:
            raw = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            # Return empty result rather than crashing if LLM returns bad JSON
            return ExtractionResult()

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

    # ── Helpers ──────────────────────────────────────────────────────────

    def _call_generator(self, prompt: str) -> str:
        """Call whatever generator interface is available."""
        gen = self._gen

        # Gemini GenerativeModel style: generate_content(prompt) -> response with .text
        if hasattr(gen, "generate_content"):
            resp = gen.generate_content(prompt)
            return resp.text

        # raglib BaseGenerator style: generate(question, context) -> str
        if hasattr(gen, "generate"):
            return gen.generate(question=prompt, context="")

        # Last resort: call the object directly
        result = gen(prompt)
        if hasattr(result, "text"):
            return result.text
        return str(result)

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove ```json ... ``` fences if present."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first line (```json or ```) and last line (```)
            inner = lines[1:] if lines[0].startswith("```") else lines
            if inner and inner[-1].strip() == "```":
                inner = inner[:-1]
            text = "\n".join(inner).strip()
        return text
