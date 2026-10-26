"""is the rulebook — it has the logic that decides how to cut a document into pieces
Chunking strategies for the real CBP/FTC knowledge base (see docs/ROADMAP.md Sec. 8.4).

Two modes, chosen per source document based on how much real structure it has:

- structured=False ("single"): the whole doc body is one section. Used for the four
  short docs, which are already single-topic (same shape as Project 1's synthetic KB).
- structured=True: `detect_sections` splits the body on structural signals found in
  the raw text (standalone short title-like lines, or "Label: rest of sentence"
  run-in labels), so a query only pulls the one topic it needs instead of a whole
  document. Used for the two long docs (cbp_internet_purchases.txt,
  cbp_prohibited_restricted_items.txt), which have no markdown headers because they
  were extracted via `innerText` (see data/kb/sources/SOURCES.md).

Within a section, `chunk_section` only splits further if the section exceeds
max_tokens, at paragraph boundaries, carrying the last paragraph of a chunk forward
as overlap into the next one (so a fact split across a paragraph boundary isn't
orphaned in only one chunk).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from ingestion.parse_raw import RawDocument

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Page chrome introduced by the innerText extraction (breadcrumb nav, accordion
# widget label, footer date) — not part of the article content, safe to drop
# mechanically rather than treat as text needing a chunking decision.
_CHROME_EXACT = {"Close all Open all"}
_LAST_MODIFIED_RE = re.compile(r"^Last Modified:", re.IGNORECASE)

# "Label: rest of the sentence" run-in headings (e.g. "Informal Entries: If the
# value..."). Excludes short inline asides that use the same punctuation shape but
# aren't real section breaks (e.g. "Note: ...", "Pitfalls: ...").
_RUN_IN_LABEL_RE = re.compile(r"^([A-Z][A-Za-z0-9 ,./'()-]{2,50}):\s+(?=[A-Z])(.*)$", re.DOTALL)
_ASIDE_LABELS = {
    "note", "hint", "reminder", "warning", "advisory", "plus", "pluses",
    "pitfall", "pitfalls", "example", "examples", "exceptions", "tip", "tips",
    "heads-up", "heads up",
}

DEFAULT_MAX_TOKENS = 450
DEFAULT_OVERLAP_PARAGRAPHS = 1


@dataclass
class Section:
    heading: str | None
    paragraphs: list[str]


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    title: str
    url: str
    retrieved_date: str
    section_heading: str | None
    chunk_index: int
    token_count: int
    text: str

def count_tokens(text: str) -> int:
    """Count tokens in a string, using the same cl100k_base encoding as OpenAI's API."""
    return len(_ENCODING.encode(text))


def _strip_chrome_lines(lines: list[str]) -> list[str]:
    """Removes webpage junk (nav breadcrumbs, "Last Modified:" footer, 
    the "Close all Open all" widget label) before any heading-detection happens: 
    so junk lines can't get mistaken for real headings.
    
    These come from the innerText extraction (see data/kb/sources/SOURCES.md), not
    from the article itself.
    """
    cleaned = list(lines)
    if cleaned and cleaned[0].startswith("Home") and "  " in cleaned[0]:
        i = 0
        while i < len(cleaned) and cleaned[i].strip():
            i += 1
        cleaned = cleaned[i:]
    while cleaned and not cleaned[-1].strip():
        cleaned = cleaned[:-1]
    if cleaned and _LAST_MODIFIED_RE.match(cleaned[-1].strip()):
        cleaned = cleaned[:-1]
    return [l for l in cleaned if l.strip() not in _CHROME_EXACT]


# A trailing conjunction/article means the line is a broken-off bullet fragment
# ("...skins, tusks, bone, feathers, or eggs; or") rather than a real title — real
# section titles are noun phrases and don't dangle like that.
_DANGLING_LAST_WORDS = {"or", "and", "the", "a", "an", "of", "to", "in", "for", "nor"}


def _is_heading_line(line: str) -> bool:
    # No length cap tight enough to admit every real heading (some run to ~70 chars)
    # also excludes every "see also" link-list item (some run past 120) without also
    # cutting a genuine sentence — so length isn't the signal; "no terminal sentence
    # punctuation" is. A stray link-list line getting misread as a heading is
    # harmless (it produces an empty section, dropped in detect_sections); a real
    # heading getting misread as body text is not (its whole section gets absorbed
    # into the wrong topic), so this errs toward over-detecting headings.
    if not line or len(line) > 140 or line[-1] in ".!?:;":
        return False
    last_word = line.split()[-1].strip(".,;:()").lower()
    return last_word not in _DANGLING_LAST_WORDS


def _lines_to_paragraphs(lines: list[str]) -> list[str]:
    """Blank-line-delimited paragraphs, with soft-wrapped lines rejoined."""
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def detect_sections(body: str) -> list[Section]:
    """Split into (heading, paragraphs) sections using two structural signals:

    - a standalone short line with no terminal sentence punctuation (a real section
      title, e.g. "Firearms"), or
    - a "Label: rest of the sentence" run-in heading at the start of a paragraph
      (e.g. "Informal Entries: If the value...").

    Both are only recognized at the start of a new paragraph (never mid-paragraph),
    to avoid misreading an ordinary short sentence as a heading. Some raw pages glue
    a heading line directly onto neighboring content with no blank line (e.g.
    "Firearms" onto the sentence below it, or a link-list line onto the next
    heading) — scanning line-by-line rather than by blank-line block is what lets
    those still be caught.
    """
    lines = _strip_chrome_lines(body.splitlines())

    sections: list[Section] = []
    heading: str | None = None
    paragraphs: list[str] = []
    current_para_lines: list[str] = []

    def finalize_para() -> None:
        if current_para_lines:
            paragraphs.append(" ".join(current_para_lines))
            current_para_lines.clear()

    def start_section(new_heading: str, initial_para: str | None = None) -> None:
        nonlocal heading, paragraphs
        finalize_para()
        if paragraphs:
            sections.append(Section(heading, paragraphs))
        heading = new_heading
        paragraphs = [initial_para] if initial_para else []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            finalize_para()
            continue

        if not current_para_lines:
            if _is_heading_line(line):
                start_section(line)
                continue
            m = _RUN_IN_LABEL_RE.match(line)
            if m:
                if m.group(1).strip().lower() not in _ASIDE_LABELS:
                    start_section(m.group(1).strip(), m.group(2).strip())
                    continue
                # An aside line ("Note: ...", "Reminder: ...") is already a
                # complete sentence on its own line in this corpus — record it as
                # its own paragraph without touching current_para_lines, so a real
                # heading glued directly onto the next line (no blank separator,
                # same artifact as above) still gets detected.
                paragraphs.append(line)
                continue

        current_para_lines.append(line)

    finalize_para()
    if paragraphs:
        sections.append(Section(heading, paragraphs))

    # A heading-like line immediately followed by another heading-like line (no
    # paragraph text in between) is never a real topic — real topics are always
    # followed by prose. Drop those empty sections rather than emit zero-content
    # chunks for them.
    return [s for s in sections if s.paragraphs]


def chunk_section(
    paragraphs: list[str],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_paragraphs: int = DEFAULT_OVERLAP_PARAGRAPHS,
) -> list[str]:
    if not paragraphs:
        return []

    if sum(count_tokens(p) for p in paragraphs) <= max_tokens:
        return ["\n\n".join(paragraphs)]

    grouped: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current and current_tokens + para_tokens > max_tokens:
            grouped.append(current)
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            current_tokens = sum(count_tokens(p) for p in current)
        current.append(para)
        current_tokens += para_tokens

    if current:
        grouped.append(current)

    return ["\n\n".join(g) for g in grouped]


def _context_header(title: str, heading: str | None) -> str:
    if heading:
        return f"[Source: {title} | Section: {heading}]"
    return f"[Source: {title}]"


def chunk_document(
    doc: RawDocument,
    structured: bool,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_paragraphs: int = DEFAULT_OVERLAP_PARAGRAPHS,
) -> list[Chunk]:
    if structured:
        sections = detect_sections(doc.body)
    else:
        lines = _strip_chrome_lines(doc.body.splitlines())
        sections = [Section(heading=None, paragraphs=_lines_to_paragraphs(lines))]

    stem = Path(doc.source_file).stem
    chunks: list[Chunk] = []
    idx = 0

    for section in sections:
        for body_text in chunk_section(section.paragraphs, max_tokens, overlap_paragraphs):
            full_text = f"{_context_header(doc.title, section.heading)}\n\n{body_text}"
            chunks.append(Chunk(
                chunk_id=f"{stem}__{idx:03d}",
                source_file=doc.source_file,
                title=doc.title,
                url=doc.url,
                retrieved_date=doc.retrieved_date,
                section_heading=section.heading,
                chunk_index=idx,
                token_count=count_tokens(full_text),
                text=full_text,
            ))
            idx += 1

    return chunks
