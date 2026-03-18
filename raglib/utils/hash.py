"""SHA-256 document change detection utilities."""
import hashlib
import json
from pathlib import Path


def content_hash(text: str) -> str:
    """Return SHA-256 hex digest of the given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class HashStore:
    """Persistent JSON store mapping doc_id → content hash."""

    def __init__(self, path: str = "./doc_hashes.json"):
        self.path = Path(path)
        self._hashes: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._hashes, indent=2), encoding="utf-8")

    def get(self, doc_id: str) -> str | None:
        return self._hashes.get(doc_id)

    def set(self, doc_id: str, hash_val: str):
        self._hashes[doc_id] = hash_val
        self.save()

    def remove(self, doc_id: str):
        self._hashes.pop(doc_id, None)
        self.save()

    def is_unchanged(self, doc_id: str, text: str) -> bool:
        """Return True if the document hasn't changed since last ingestion."""
        return self._hashes.get(doc_id) == content_hash(text)

    def all_doc_ids(self) -> set[str]:
        return set(self._hashes.keys())
