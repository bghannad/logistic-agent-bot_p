"""Parse the raw KB .txt files in data/kb/raw/ into header metadata + body text.

Each raw file has a "KEY: value" header block (source/title/url/retrieved date/etc.,
see data/kb/sources/SOURCES.md for what each field means) followed by a "---" line,
then the verbatim page text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawDocument:
    source_file: str
    header: dict[str, str]
    body: str

    @property
    def title(self) -> str:
        return self.header.get("TITLE", self.source_file)

    @property
    def url(self) -> str:
        return self.header.get("URL", "")

    @property
    def retrieved_date(self) -> str:
        return self.header.get("RETRIEVED", "")


def parse_raw_file(path: Path) -> RawDocument:
    text = path.read_text(encoding="utf-8")
    header_block, _, body = text.partition("\n---\n")
    if not _:
        raise ValueError(f"{path}: no '---' header/body separator found")

    header: dict[str, str] = {}
    for line in header_block.splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        if sep:
            header[key.strip()] = value.strip()

    return RawDocument(source_file=path.name, header=header, body=body.strip("\n"))
