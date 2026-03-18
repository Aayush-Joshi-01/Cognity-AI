"""NLPExtractor — full NLP extraction pipeline ported from hybrid_rag/nlp_processor.py."""
from collections import defaultdict
from typing import Optional

from raglib.extractors.base import BaseExtractor
from raglib.models.knowledge import Entity, Relation, ExtractionResult


# ── Mapping tables (identical to nlp_processor.py) ──────────────────────────

SPACY_TO_TYPE = {
    "PERSON": "Person", "ORG": "Organization", "GPE": "Location",
    "LOC": "Location", "FAC": "Location", "NORP": "Group",
    "PRODUCT": "Product", "EVENT": "Event", "WORK_OF_ART": "CreativeWork",
    "LAW": "Law", "LANGUAGE": "Language", "DATE": "Temporal",
    "TIME": "Temporal", "MONEY": "Monetary", "QUANTITY": "Quantity",
    "CARDINAL": "Quantity", "ORDINAL": "Quantity", "PERCENT": "Quantity",
}

DEP_TO_RELATION = {
    ("nsubj", "dobj"): "ACTS_ON",
    ("nsubj", "attr"): "IS_A",
    ("nsubj", "pobj"): "RELATES_TO",
    ("nsubjpass", "agent"): "ACTED_ON_BY",
    ("compound",): "PART_OF",
    ("appos",): "ALSO_KNOWN_AS",
    ("amod",): "HAS_PROPERTY",
    ("poss",): "BELONGS_TO",
}


class NLPExtractor(BaseExtractor):
    """Extracts entities and relations via spaCy NLP pipeline.

    This is a standalone port of ``NLPProcessor.process()`` and all its private
    methods from ``hybrid_rag/nlp_processor.py``.
    """

    def __init__(self, nlp_model, config=None):
        """
        Args:
            nlp_model: A loaded spaCy Language object.
            config: Optional config object (currently unused; reserved for future tuning).
        """
        self.nlp = nlp_model
        self.config = config

    # ── BaseExtractor interface ──────────────────────────────────────────

    def extract(self, text: str, source_id: str = "") -> ExtractionResult:
        """Full NLP extraction: NER + dependency triples + coref.

        Args:
            text: Input text to process.
            source_id: Document/source identifier attached to extracted items.

        Returns:
            ExtractionResult with deduplicated entities and relations.
        """
        doc = self.nlp(text)

        entities = self._extract_entities(doc, source_id)
        relations = self._extract_dep_relations(doc, source_id)
        svo_relations = self._extract_svo_triples(doc, source_id)
        relations.extend(svo_relations)
        entities, relations = self._resolve_coreferences(doc, entities, relations)
        np_entities = self._extract_noun_phrase_entities(doc, source_id)
        entities.extend(np_entities)

        entities = self._deduplicate_entities(entities)
        relations = self._deduplicate_relations(relations)

        return ExtractionResult(entities=entities, relations=relations)

    # ── NER Extraction ───────────────────────────────────────────────────

    def _extract_entities(self, doc, source_id: str) -> list:
        entities = []
        seen = set()
        for ent in doc.ents:
            name = ent.text.strip()
            if len(name) < 2 or name.lower() in seen:
                continue
            seen.add(name.lower())

            etype = SPACY_TO_TYPE.get(ent.label_, "Other")
            if etype in ("Quantity", "Temporal", "Monetary") and len(name) < 4:
                continue

            entities.append(Entity(
                name=name.title() if etype != "Temporal" else name,
                entity_type=etype,
                description=(
                    f"{etype} entity extracted via NER from context: "
                    f"'...{doc.text[max(0, ent.start_char - 30):ent.end_char + 30]}...'"
                ),
                source_id=source_id,
                confidence=0.85,
                extraction_method="nlp",
                mentions=sum(1 for e2 in doc.ents if e2.text.lower() == name.lower()),
            ))
        return entities

    # ── Dependency Relation Extraction ───────────────────────────────────

    def _extract_dep_relations(self, doc, source_id: str) -> list:
        relations = []
        for sent in doc.sents:
            ents_in_sent = [
                e for e in doc.ents
                if e.start >= sent.start and e.end <= sent.end
            ]
            if len(ents_in_sent) < 2:
                continue

            for i, e1 in enumerate(ents_in_sent):
                for e2 in ents_in_sent[i + 1:]:
                    rel = self._find_dep_relation(e1.root, e2.root, sent)
                    if rel:
                        relations.append(Relation(
                            source_entity=e1.text.strip().title(),
                            relation_type=rel,
                            target_entity=e2.text.strip().title(),
                            description=sent.text.strip(),
                            source_id=source_id,
                            confidence=0.75,
                            extraction_method="nlp",
                        ))
        return relations

    def _find_dep_relation(self, token1, token2, sent) -> Optional[str]:
        if token2.head == token1:
            return self._dep_to_relation(token2.dep_)
        if token1.head == token2:
            return self._dep_to_relation(token1.dep_)

        if token1.head == token2.head:
            verb = token1.head
            if verb.pos_ == "VERB":
                return verb.lemma_.upper().replace(" ", "_")

        ancestor = token1.head
        for _ in range(4):
            if ancestor == token2 or ancestor == token2.head:
                return self._dep_to_relation(token1.dep_)
            if ancestor.head == ancestor:
                break
            ancestor = ancestor.head

        return None

    @staticmethod
    def _dep_to_relation(dep: str) -> str:
        mapping = {
            "nsubj": "SUBJECT_OF", "nsubjpass": "OBJECT_OF",
            "dobj": "ACTS_ON", "pobj": "RELATES_TO",
            "attr": "IS_A", "appos": "ALSO_KNOWN_AS",
            "agent": "ACTED_BY", "prep": "ASSOCIATED_WITH",
            "compound": "PART_OF", "amod": "HAS_PROPERTY",
            "poss": "BELONGS_TO", "conj": "AND",
        }
        return mapping.get(dep, "RELATES_TO")

    # ── SVO Triple Extraction ────────────────────────────────────────────

    def _extract_svo_triples(self, doc, source_id: str) -> list:
        relations = []
        for sent in doc.sents:
            for token in sent:
                if token.pos_ != "VERB":
                    continue

                subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
                objects = [c for c in token.children if c.dep_ in ("dobj", "attr", "pobj")]

                for prep in [c for c in token.children if c.dep_ == "prep"]:
                    objects.extend(c for c in prep.children if c.dep_ == "pobj")

                for subj in subjects:
                    subj_text = self._get_compound_text(subj)
                    for obj in objects:
                        obj_text = self._get_compound_text(obj)
                        if len(subj_text) > 1 and len(obj_text) > 1:
                            rel_type = token.lemma_.upper().replace(" ", "_")
                            rel_type = self._normalize_verb_relation(rel_type)
                            relations.append(Relation(
                                source_entity=subj_text.title(),
                                relation_type=rel_type,
                                target_entity=obj_text.title(),
                                description=sent.text.strip(),
                                source_id=source_id,
                                confidence=0.7,
                                extraction_method="nlp_svo",
                            ))
        return relations

    @staticmethod
    def _get_compound_text(token) -> str:
        compounds = [c for c in token.children if c.dep_ in ("compound", "amod", "flat")]
        parts = sorted([token] + compounds, key=lambda t: t.i)
        return " ".join(t.text for t in parts).strip()

    @staticmethod
    def _normalize_verb_relation(verb: str) -> str:
        verb_map = {
            "BE": "IS_A", "HAVE": "HAS", "FOUND": "FOUNDED_BY",
            "CREATE": "CREATED", "DEVELOP": "DEVELOPED", "BUILD": "BUILT",
            "LEAD": "LED_BY", "OWN": "OWNS", "ACQUIRE": "ACQUIRED",
            "INVEST": "INVESTED_IN", "LOCATE": "LOCATED_IN",
            "BASE": "BASED_IN", "HEADQUARTER": "HEADQUARTERED_IN",
            "LAUNCH": "LAUNCHED", "RELEASE": "RELEASED",
            "USE": "USES", "WORK": "WORKS_AT", "SERVE": "SERVES_AS",
            "RAISE": "RAISED", "FOCUS": "FOCUSES_ON",
        }
        return verb_map.get(verb, verb)

    # ── Coreference Resolution ────────────────────────────────────────────

    def _resolve_coreferences(self, doc, entities: list, relations: list) -> tuple:
        ent_positions = []
        for ent in doc.ents:
            ent_positions.append({
                "text": ent.text, "start": ent.start, "label": ent.label_,
            })

        pronoun_map = {}
        personal = {"he", "she", "they", "him", "her", "them", "his", "their", "its"}
        for token in doc:
            if token.text.lower() in personal:
                best = None
                for ep in reversed(ent_positions):
                    if ep["start"] < token.i:
                        if token.text.lower() in ("he", "she", "him", "her", "his"):
                            if ep["label"] == "PERSON":
                                best = ep["text"]
                                break
                        elif token.text.lower() in ("it", "its"):
                            if ep["label"] in ("ORG", "PRODUCT", "GPE"):
                                best = ep["text"]
                                break
                        else:
                            best = ep["text"]
                            break
                if best:
                    pronoun_map[token.i] = best

        for rel in relations:
            if rel.source_entity.lower() in personal:
                for tidx, resolved in pronoun_map.items():
                    rel.source_entity = resolved.title()
                    break
            if rel.target_entity.lower() in personal:
                for tidx, resolved in pronoun_map.items():
                    rel.target_entity = resolved.title()
                    break

        return entities, relations

    # ── Noun Phrase Entity Extraction ────────────────────────────────────

    def _extract_noun_phrase_entities(self, doc, source_id: str) -> list:
        entities = []
        existing_names = {e.text.lower() for e in doc.ents}

        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            if (
                len(text) < 3
                or text.lower() in existing_names
                or chunk.root.pos_ in ("PRON", "DET")
            ):
                continue

            has_modifier = any(t.dep_ in ("amod", "compound", "nmod") for t in chunk)
            if not has_modifier and len(text.split()) < 2:
                continue

            entities.append(Entity(
                name=text.title(),
                entity_type="Concept",
                description=f"Noun phrase from: '{chunk.sent.text.strip()[:100]}'",
                source_id=source_id,
                confidence=0.6,
                extraction_method="nlp_np",
            ))
        return entities

    # ── Deduplication ────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate_entities(entities: list) -> list:
        seen: dict = {}
        for e in entities:
            key = e.name.lower().strip()
            if key not in seen or e.confidence > seen[key].confidence:
                if key in seen:
                    e.mentions += seen[key].mentions
                    if seen[key].extraction_method != e.extraction_method:
                        e.extraction_method = "merged"
                seen[key] = e
        return list(seen.values())

    @staticmethod
    def _deduplicate_relations(relations: list) -> list:
        seen: dict = {}
        for r in relations:
            key = f"{r.source_entity.lower()}|{r.relation_type}|{r.target_entity.lower()}"
            if key not in seen or r.confidence > seen[key].confidence:
                if key in seen:
                    r.weight += seen[key].weight
                seen[key] = r
        return list(seen.values())
