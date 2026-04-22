"""
chunk_docs.py — Section-based document chunker for VP RAG Eval

Splits each cooking technique document into chunks at section headers.
A section header is an ALL CAPS line ending with a colon (e.g. PURPOSE:, HOW TO BLANCH:)
Parenthetical content in headers is allowed (e.g. TEMPERING TEMPERATURES (dark chocolate):)

Usage:
    python3 scripts/chunk_docs.py

Output:
    data/chunks.json — all chunks with metadata, ready to load into ChromaDB
"""

import json
import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DOCS_DIR = ROOT / "data" / "docs"
OUTPUT   = ROOT / "data" / "chunks.json"

MIN_CHUNK_LINES = 2  # skip chunks with fewer than this many content lines


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_section_header(line: str) -> bool:
    """
    A section header is a line that:
    1. Ends with a colon (nothing after it on the line)
    2. Has only uppercase letters when parenthetical content is removed
    Examples: PURPOSE:  HOW TO BLANCH:  TEMPERING TEMPERATURES (dark chocolate):
    Not a header: TEST: Dip a knife tip...  (has content after colon)
    """
    line = line.strip()
    if not line.endswith(":"):
        return False
    text = line[:-1].strip()  # remove trailing colon
    # Remove parenthetical content e.g. "(dark chocolate)" "(easier)"
    text_no_parens = re.sub(r"\([^)]*\)", "", text).strip()
    if len(text_no_parens) < 2:
        return False
    # All alphabetic characters must be uppercase
    alpha_chars = [c for c in text_no_parens if c.isalpha()]
    if not alpha_chars:
        return False
    return all(c.isupper() for c in alpha_chars)


def extract_topic(lines: list[str]) -> str:
    """Extract topic from TOPIC: line at top of document."""
    for line in lines[:3]:
        if line.strip().startswith("TOPIC:"):
            return line.split(":", 1)[1].strip()
    return "Unknown"


def chunk_document(doc_path: Path) -> list[dict]:
    """Split a document into section-based chunks."""
    lines   = doc_path.read_text(encoding="utf-8").splitlines()
    topic   = extract_topic(lines)
    doc_id  = doc_path.stem  # e.g. doc_01_blanching

    chunks          = []
    current_section = "INTRODUCTION"
    current_lines   = []

    for line in lines:
        if line.strip().startswith("TOPIC:"):
            continue  # skip the topic line — it's metadata, not content

        if is_section_header(line):
            # Save current section as a chunk (if enough content)
            if len([l for l in current_lines if l.strip()]) >= MIN_CHUNK_LINES:
                chunks.append({
                    "doc_id":   doc_id,
                    "topic":    topic,
                    "section":  current_section,
                    "content":  "\n".join(current_lines).strip(),
                    "chunk_id": f"{doc_id}__{current_section.replace(' ', '_').lower()}",
                })
            # Start new section
            current_section = line.strip().rstrip(":")
            current_lines   = []
        else:
            current_lines.append(line)

    # Save last section
    if len([l for l in current_lines if l.strip()]) >= MIN_CHUNK_LINES:
        chunks.append({
            "doc_id":   doc_id,
            "topic":    topic,
            "section":  current_section,
            "content":  "\n".join(current_lines).strip(),
            "chunk_id": f"{doc_id}__{current_section.replace(' ', '_').lower()}",
        })

    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    doc_files = sorted(DOCS_DIR.glob("*.txt"))
    all_chunks = []

    print(f"\n✂️  VP RAG Eval — Document Chunker")
    print(f"   Strategy : Section-based (ALL CAPS headers)")
    print(f"   Docs dir : {DOCS_DIR}")
    print(f"   Documents: {len(doc_files)}\n")

    for doc_path in doc_files:
        chunks = chunk_document(doc_path)
        all_chunks.extend(chunks)
        print(f"  {doc_path.name:<45} → {len(chunks)} chunks")

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*55}")
    print(f"  Total documents : {len(doc_files)}")
    print(f"  Total chunks    : {len(all_chunks)}")
    print(f"  Avg per doc     : {len(all_chunks) / len(doc_files):.1f}")
    print(f"{'='*55}")
    print(f"\n✅ Chunks saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
