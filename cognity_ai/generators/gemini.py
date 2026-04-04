"""GeminiGenerator — default generator using Google Gemini models (google-genai SDK).

Also exposes augment_extraction() and summarize_community() so that the
same Gemini client can serve both generation and knowledge-graph extraction
tasks without requiring a separate GeminiExtractor instance.
"""
import json
import time

from cognity_ai.generators.base import BaseGenerator, GENERATION_PROMPT


# ── Extraction prompts ───────────────────────────────────────────────────────

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


def _build_client(config_or_key=None, *, api_key=None, project_id=None,
                  location="us-central1", use_vertexai=False, timeout=120):
    """Build a google.genai Client from a GeminiConfig, an api_key string, or env vars."""
    from google import genai
    from google.genai import types as gentypes

    http_opts = gentypes.HttpOptions(timeout=timeout)

    if config_or_key is not None and not isinstance(config_or_key, str):
        cfg = config_or_key
        if getattr(cfg, "use_vertexai", False) and getattr(cfg, "project_id", ""):
            return genai.Client(
                vertexai=True,
                project=cfg.project_id,
                location=getattr(cfg, "location", "us-central1"),
                http_options=http_opts,
            )
        key = getattr(cfg, "api_key", "") or None
        if key:
            return genai.Client(api_key=key, http_options=http_opts)
        return genai.Client(http_options=http_opts)

    if isinstance(config_or_key, str):
        api_key = api_key or config_or_key

    if use_vertexai and project_id:
        return genai.Client(vertexai=True, project=project_id,
                            location=location, http_options=http_opts)
    if api_key:
        return genai.Client(api_key=api_key, http_options=http_opts)
    return genai.Client(http_options=http_opts)


class GeminiGenerator(BaseGenerator):
    """Generate answers using Google Gemini via the google-genai SDK.

    Accepts a ``GeminiConfig`` object (as used by the factory), an explicit
    ``api_key`` string, or no arguments at all — in which case the SDK
    automatically reads ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``) from the
    environment.

    Optionally provide ``project_id`` + ``use_vertexai=True`` for project-based
    Vertex AI access instead of a plain API key.

    Examples::

        # Factory / config-object usage:
        gen = GeminiGenerator(cfg.gemini)

        # Explicit key:
        gen = GeminiGenerator(api_key="AIza...")

        # Vertex AI / project-based:
        gen = GeminiGenerator(project_id="my-gcp-project", use_vertexai=True)

        # Env-var auto-load (GOOGLE_API_KEY):
        gen = GeminiGenerator()
    """

    def __init__(
        self,
        config=None,
        *,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        extraction_temperature: float | None = None,
        rpm_limit: int | None = None,
        project_id: str | None = None,
        location: str | None = None,
        use_vertexai: bool = False,
    ):
        cfg_model = "gemini-2.0-flash"
        cfg_temp = 0.1
        cfg_ext_temp = 0.0
        cfg_rpm = 15
        cfg_timeout = 120

        if config is not None and not isinstance(config, str):
            cfg_model = getattr(config, "model", cfg_model)
            cfg_temp = getattr(config, "temperature", cfg_temp)
            cfg_ext_temp = getattr(config, "extraction_temperature", cfg_ext_temp)
            cfg_rpm = getattr(config, "rpm_limit", cfg_rpm)
            cfg_timeout = getattr(config, "timeout", cfg_timeout)
        elif isinstance(config, str):
            api_key = api_key or config
            config = None

        self._model = model or cfg_model
        self._temperature = temperature if temperature is not None else cfg_temp
        self._extraction_temperature = (
            extraction_temperature if extraction_temperature is not None else cfg_ext_temp
        )
        self._rpm_limit = rpm_limit if rpm_limit is not None else cfg_rpm
        self._last_call = 0.0
        self._client = _build_client(
            config,
            api_key=api_key,
            project_id=project_id,
            location=location or "us-central1",
            use_vertexai=use_vertexai,
            timeout=cfg_timeout,
        )

    def _rate_limit(self):
        gap = 60.0 / self._rpm_limit
        elapsed = time.time() - self._last_call
        if elapsed < gap:
            time.sleep(gap - elapsed)
        self._last_call = time.time()

    def _gen_config(self, temperature: float, json_mode: bool = False):
        from google.genai import types as gentypes
        kwargs = {"temperature": temperature}
        if json_mode:
            kwargs["response_mime_type"] = "application/json"
        return gentypes.GenerateContentConfig(**kwargs)

    # ── BaseGenerator interface ──────────────────────────────────────────

    def generate(self, question: str, context: str) -> str:
        """Generate an answer for question using the provided context string."""
        self._rate_limit()
        if question:
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = context
        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._gen_config(self._temperature),
        )
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
        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._gen_config(self._temperature),
        )
        return resp.text

    # ── Knowledge-graph helpers ──────────────────────────────────────────

    def augment_extraction(self, text: str, existing, source_id: str = ""):
        """Find entities/relations that NLP missed."""
        from cognity_ai.models.knowledge import Entity, Relation, ExtractionResult

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
        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._gen_config(self._extraction_temperature, json_mode=True),
        )
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
        """Return a title/summary dict for a graph community."""
        self._rate_limit()
        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            entities=", ".join(entity_names),
            relations="\n".join(f"- {r}" for r in relation_descriptions[:20]),
        )
        resp = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._gen_config(self._extraction_temperature, json_mode=True),
        )
        return json.loads(resp.text)
