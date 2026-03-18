"""CSV and TSV loader with auto-delimiter detection."""
from __future__ import annotations
import csv
import io
import uuid
from pathlib import Path

from raglib.loaders.base import BaseLoader
from raglib.models.document import Document


class CsvLoader(BaseLoader):
    """
    Loads CSV/TSV files as a single Document with readable text.
    Auto-detects delimiter (comma, tab, semicolon, pipe).
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".csv", ".tsv"]

    def load(self, path: str) -> list[Document]:
        p = Path(path)
        raw = p.read_text(encoding="utf-8", errors="replace")
        doc_id = p.stem + "_" + uuid.uuid4().hex[:8]

        delimiter = self._detect_delimiter(raw, p.suffix.lower())
        text = self._csv_to_text(raw, delimiter)

        return [
            Document(
                doc_id=doc_id,
                text=text,
                source_path=str(p.resolve()),
                source_name=p.name,
                loader="CsvLoader",
                file_extension=p.suffix.lower(),
                file_size_bytes=p.stat().st_size,
                page_count=1,
                page_map=[{"page_num": 1, "start_char": 0, "end_char": len(text), "heading": ""}],
            )
        ]

    def _detect_delimiter(self, raw: str, suffix: str) -> str:
        """Auto-detect delimiter. TSV files default to tab."""
        if suffix == ".tsv":
            return "\t"

        # Use csv.Sniffer on a sample
        sample = raw[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            return dialect.delimiter
        except csv.Error:
            pass

        # Fallback: pick whichever delimiter appears most in the first line
        first_line = raw.split("\n", 1)[0] if "\n" in raw else raw
        counts = {d: first_line.count(d) for d in (",", "\t", ";", "|")}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","

    def _csv_to_text(self, raw: str, delimiter: str) -> str:
        """Convert CSV rows to a human-readable tabular text block."""
        reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
        rows = list(reader)

        if not rows:
            return ""

        lines = []
        header = rows[0]
        # Header context line
        lines.append("Columns: " + ", ".join(header))
        lines.append("")  # blank line separator

        for row in rows:
            parts = []
            for col, val in zip(header, row):
                parts.append(f"{col}: {val}")
            # Pad if row is longer than header
            for val in row[len(header):]:
                parts.append(val)
            lines.append(" | ".join(parts))

        return "\n".join(lines)
