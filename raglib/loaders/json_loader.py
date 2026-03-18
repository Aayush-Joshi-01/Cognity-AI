"""JSON and YAML file loaders that flatten nested structures to readable text."""
from __future__ import annotations
import json
import uuid
from pathlib import Path

from raglib.loaders.base import BaseLoader
from raglib.models.document import Document

_MAX_DEPTH = 4


def _flatten(obj, prefix: str = "", depth: int = 0, lines: list[str] | None = None) -> list[str]:
    """Recursively flatten a nested dict/list into 'key: value' lines."""
    if lines is None:
        lines = []
    if depth >= _MAX_DEPTH:
        lines.append(f"{prefix}: {repr(obj)}")
        return lines

    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else str(k)
            _flatten(v, full_key, depth + 1, lines)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            full_key = f"{prefix}[{i}]"
            _flatten(v, full_key, depth + 1, lines)
    else:
        lines.append(f"{prefix}: {obj}")
    return lines


class JsonLoader(BaseLoader):
    """Loads .json files by flattening nested structure to key: value text."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".json"]

    def load(self, path: str) -> list[Document]:
        p = Path(path)
        raw = p.read_text(encoding="utf-8", errors="replace")
        doc_id = p.stem + "_" + uuid.uuid4().hex[:8]

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            # Return raw text on parse error with error annotation
            text = f"[JSON parse error: {exc}]\n\n{raw}"
        else:
            lines = _flatten(data)
            text = "\n".join(lines)

        return [
            Document(
                doc_id=doc_id,
                text=text,
                source_path=str(p.resolve()),
                source_name=p.name,
                loader="JsonLoader",
                file_extension=".json",
                file_size_bytes=p.stat().st_size,
                page_count=1,
                page_map=[{"page_num": 1, "start_char": 0, "end_char": len(text), "heading": ""}],
            )
        ]


class YamlLoader(BaseLoader):
    """Loads .yaml/.yml files by flattening nested structure to key: value text."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".yaml", ".yml"]

    def load(self, path: str) -> list[Document]:
        try:
            import yaml  # type: ignore
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML files. "
                "Install it with: pip install pyyaml"
            )

        p = Path(path)
        raw = p.read_text(encoding="utf-8", errors="replace")
        doc_id = p.stem + "_" + uuid.uuid4().hex[:8]

        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            text = f"[YAML parse error: {exc}]\n\n{raw}"
        else:
            if data is None:
                text = ""
            else:
                lines = _flatten(data)
                text = "\n".join(lines)

        return [
            Document(
                doc_id=doc_id,
                text=text,
                source_path=str(p.resolve()),
                source_name=p.name,
                loader="YamlLoader",
                file_extension=p.suffix.lower(),
                file_size_bytes=p.stat().st_size,
                page_count=1,
                page_map=[{"page_num": 1, "start_char": 0, "end_char": len(text), "heading": ""}],
            )
        ]
