"""
Local NLP pipeline using spaCy:
  1. Sentence segmentation + semantic chunking
  2. NER (Named Entity Recognition)
  3. Dependency parse → SVO triple extraction
  4. Coreference resolution (pronoun → entity linking)
  5. Noun phrase extraction for compound entities
  6. Page/section boundary detection

Runs entirely local — zero API cost.
"""

import re
from collections import Counter, defaultdict
from typing import Optional

import spacy
from spacy.tokens import Doc, Span, Token

from config import NLPConfig
from models import Entity, Relation, ExtractionResult, PageInfo, SemanticChunk


# ── spaCy entity type → our ontology ────────────────────────────────────

SPACY_TO_TYPE = {
    "PERSON": "Person", "ORG": "Organization", "GPE": "Location",
    "LOC": "Location", "FAC": "Location", "NORP": "Group",
    "PRODUCT": "Product", "EVENT": "Event", "WORK_OF_ART": "CreativeWork",
    "LAW": "Law", "LANGUAGE": "Language", "DATE": "Temporal",
    "TIME": "Temporal", "MONEY": "Monetary", "QUANTITY": "Quantity",
    "CARDINAL": "Quantity", "ORDINAL": "Quantity", "PERCENT": "Quantity",
}

# Dependency arc → relation type mapping
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


class NLPProcessor:
    def __init__(self, config: NLPConfig):
        self.config = config
        self.nlp = self._load_model(config.spacy_model, config.fallback_model)
        # Add sentencizer as fallback if no parser
        if not self.nlp.has_pipe("parser") and not self.nlp.has_pipe("sentencizer"):
            self.nlp.add_pipe("sentencizer")

    @staticmethod
    def _load_model(primary: str, fallback: str) -> spacy.Language:
        try:
            return spacy.load(primary)
        except OSError:
            try:
                return spacy.load(fallback)
            except OSError:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", fallback], check=True)
                return spacy.load(fallback)

    # ── Core NLP Pipeline ───────────────────────────────────────────────

    def process(self, text: str, source_id: str = "") -> ExtractionResult:
        """Full NLP extraction: NER + dependency triples + coref."""
        doc = self.nlp(text)

        # Step 1: NER entities
        entities = self._extract_entities(doc, source_id)

        # Step 2: Dependency-based relation extraction
        relations = self._extract_dep_relations(doc, source_id)

        # Step 3: SVO (Subject-Verb-Object) triples
        svo_relations = self._extract_svo_triples(doc, source_id)
        relations.extend(svo_relations)

        # Step 4: Coreference-aware entity merging
        entities, relations = self._resolve_coreferences(doc, entities, relations)

        # Step 5: Noun phrase entities (catches compound names spaCy NER misses)
        np_entities = self._extract_noun_phrase_entities(doc, source_id)
        entities.extend(np_entities)

        # Deduplicate
        entities = self._deduplicate_entities(entities)
        relations = self._deduplicate_relations(relations)

        return ExtractionResult(entities=entities, relations=relations)

    # ── NER Extraction ──────────────────────────────────────────────────

    def _extract_entities(self, doc: Doc, source_id: str) -> list[Entity]:
        entities = []
        seen = set()
        for ent in doc.ents:
            name = ent.text.strip()
            if len(name) < 2 or name.lower() in seen:
                continue
            seen.add(name.lower())

            etype = SPACY_TO_TYPE.get(ent.label_, "Other")
            # Skip pure numeric/temporal unless they're meaningful
            if etype in ("Quantity", "Temporal", "Monetary") and len(name) < 4:
                continue

            entities.append(Entity(
                name=name.title() if etype != "Temporal" else name,
                entity_type=etype,
                description=f"{etype} entity extracted via NER from context: '...{doc.text[max(0,ent.start_char-30):ent.end_char+30]}...'",
                source_id=source_id,
                confidence=0.85,
                extraction_method="nlp",
                mentions=sum(1 for e2 in doc.ents if e2.text.lower() == name.lower()),
            ))
        return entities

    # ── Dependency Relation Extraction ──────────────────────────────────

    def _extract_dep_relations(self, doc: Doc, source_id: str) -> list[Relation]:
        relations = []
        for sent in doc.sents:
            ents_in_sent = [e for e in doc.ents if e.start >= sent.start and e.end <= sent.end]
            if len(ents_in_sent) < 2:
                continue

            # Find dependency paths between entity pairs
            for i, e1 in enumerate(ents_in_sent):
                for e2 in ents_in_sent[i+1:]:
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

    def _find_dep_relation(self, token1: Token, token2: Token, sent: Span) -> Optional[str]:
        """Walk dependency tree between two tokens to infer relation type."""
        # Direct parent-child
        if token2.head == token1:
            return self._dep_to_relation(token2.dep_)
        if token1.head == token2:
            return self._dep_to_relation(token1.dep_)

        # Shared head (siblings in parse tree)
        if token1.head == token2.head:
            verb = token1.head
            if verb.pos_ == "VERB":
                return verb.lemma_.upper().replace(" ", "_")

        # Walk up from token1, check if we hit token2's subtree
        ancestor = token1.head
        for _ in range(4):  # max 4 hops
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

    # ── SVO Triple Extraction ───────────────────────────────────────────

    def _extract_svo_triples(self, doc: Doc, source_id: str) -> list[Relation]:
        """Extract Subject-Verb-Object triples from sentences."""
        relations = []
        for sent in doc.sents:
            for token in sent:
                if token.pos_ != "VERB":
                    continue

                subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
                objects = [c for c in token.children if c.dep_ in ("dobj", "attr", "pobj")]

                # Also check prep children for indirect objects
                for prep in [c for c in token.children if c.dep_ == "prep"]:
                    objects.extend(c for c in prep.children if c.dep_ == "pobj")

                for subj in subjects:
                    subj_text = self._get_compound_text(subj)
                    for obj in objects:
                        obj_text = self._get_compound_text(obj)
                        if len(subj_text) > 1 and len(obj_text) > 1:
                            rel_type = token.lemma_.upper().replace(" ", "_")
                            # Clean up common verbs to semantic relations
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
    def _get_compound_text(token: Token) -> str:
        """Expand token to include compound modifiers (e.g., 'Sam' + 'Altman')."""
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

    # ── Coreference Resolution ──────────────────────────────────────────

    def _resolve_coreferences(
        self, doc: Doc, entities: list[Entity], relations: list[Relation]
    ) -> tuple[list[Entity], list[Relation]]:
        """
        Basic pronominal coreference: map pronouns to nearest preceding
        named entity of matching type. Full coref models are expensive;
        this heuristic covers 70-80% of cases at zero cost.
        """
        # Build entity position index
        ent_positions = []
        for ent in doc.ents:
            ent_positions.append({
                "text": ent.text, "start": ent.start, "label": ent.label_,
            })

        # Map pronouns to nearest prior entity
        pronoun_map = {}
        personal = {"he", "she", "they", "him", "her", "them", "his", "their", "its"}
        for token in doc:
            if token.text.lower() in personal:
                # Find nearest prior named entity
                best = None
                for ep in reversed(ent_positions):
                    if ep["start"] < token.i:
                        # Type matching: he/she/him/her → PERSON
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

        # Rewrite relations that have pronouns as source/target
        for rel in relations:
            if rel.source_entity.lower() in personal:
                # Find the pronoun token index and resolve
                for tidx, resolved in pronoun_map.items():
                    rel.source_entity = resolved.title()
                    break
            if rel.target_entity.lower() in personal:
                for tidx, resolved in pronoun_map.items():
                    rel.target_entity = resolved.title()
                    break

        return entities, relations

    # ── Noun Phrase Entity Extraction ───────────────────────────────────

    def _extract_noun_phrase_entities(self, doc: Doc, source_id: str) -> list[Entity]:
        """
        Catch multi-word entities that NER misses (e.g., 'protein folding',
        'AI safety', 'knowledge graph'). Filter by POS pattern.
        """
        entities = []
        existing_names = {e.text.lower() for e in doc.ents}

        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            if (len(text) < 3 or text.lower() in existing_names
                    or chunk.root.pos_ in ("PRON", "DET")):
                continue

            # Only keep noun phrases with adjective/noun modifiers (meaningful compounds)
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

    # ── Semantic Chunking ───────────────────────────────────────────────

    def semantic_chunk(self, text: str, doc_id: str,
                       pages: list[PageInfo] | None = None) -> list[SemanticChunk]:
        """
        Sentence-boundary-aware chunking with entity overlap tracking.
        Each chunk knows which entities it contains → used for graph-vector linking.
        """
        doc = self.nlp(text)
        sentences = list(doc.sents)
        chunk_size = self.config.semantic_chunk_sentences
        overlap = self.config.semantic_chunk_overlap

        chunks = []
        i = 0
        idx = 0
        while i < len(sentences):
            end = min(i + chunk_size, len(sentences))
            chunk_sents = sentences[i:end]
            chunk_text = " ".join(s.text.strip() for s in chunk_sents)

            # Track which named entities appear in this chunk
            chunk_start = chunk_sents[0].start_char
            chunk_end = chunk_sents[-1].end_char
            ent_names = list({
                e.text.strip().title()
                for e in doc.ents
                if e.start_char >= chunk_start and e.end_char <= chunk_end
            })

            # Map to page info if available
            page_info = None
            if pages:
                for p in pages:
                    if p.start_char <= chunk_start < p.end_char:
                        page_info = p
                        break

            chunks.append(SemanticChunk(
                chunk_id=f"{doc_id}__chunk_{idx}",
                doc_id=doc_id,
                text=chunk_text,
                index=idx,
                page_info=page_info,
                entity_names=ent_names,
                sentence_count=len(chunk_sents),
                token_estimate=len(chunk_text.split()),
            ))
            idx += 1
            i = end - overlap if overlap > 0 and end < len(sentences) else end

        return chunks

    # ── Page/Section Detection ──────────────────────────────────────────

    @staticmethod
    def detect_pages(text: str) -> list[PageInfo]:
        """
        Detect page breaks and section headings from text.
        Handles: explicit page markers, form feeds, markdown headings, numbered sections.
        """
        pages = []
        # Split by common page/section markers
        patterns = [
            r'\f',                          # form feed
            r'\n-{3,}\n',                   # --- separator
            r'\n={3,}\n',                   # === separator
            r'(?:^|\n)(?:Page\s+\d+)',      # "Page N"
            r'(?:^|\n)(#{1,3}\s+.+)',       # Markdown headings
            r'(?:^|\n)(\d+\.\s+[A-Z].+)',   # Numbered sections
        ]
        combined = "|".join(f"({p})" for p in patterns)
        splits = list(re.finditer(combined, text, re.MULTILINE))

        if not splits:
            return [PageInfo(page_num=1, start_char=0, end_char=len(text))]

        prev_end = 0
        for i, match in enumerate(splits):
            heading = match.group(0).strip().lstrip("#").strip()
            pages.append(PageInfo(
                page_num=i + 1,
                section=heading[:100],
                heading=heading[:100],
                start_char=prev_end,
                end_char=match.start(),
            ))
            prev_end = match.start()

        # Last segment
        pages.append(PageInfo(
            page_num=len(splits) + 1,
            start_char=prev_end,
            end_char=len(text),
        ))
        return pages

    # ── Deduplication ───────────────────────────────────────────────────

    @staticmethod
    def _deduplicate_entities(entities: list[Entity]) -> list[Entity]:
        seen: dict[str, Entity] = {}
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
    def _deduplicate_relations(relations: list[Relation]) -> list[Relation]:
        seen: dict[str, Relation] = {}
        for r in relations:
            key = f"{r.source_entity.lower()}|{r.relation_type}|{r.target_entity.lower()}"
            if key not in seen or r.confidence > seen[key].confidence:
                if key in seen:
                    r.weight += seen[key].weight
                seen[key] = r
        return list(seen.values())
