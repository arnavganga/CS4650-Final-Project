#!/usr/bin/env python3
"""
Improved ToS Clause Extractor
----------------------------------
- Better HTML normalization with improved list handling
- Preserves context by keeping list items with their parent text
- Clause splitting on sentence boundaries
- Filters out headings/junk (no verbs, too short, all caps)
- Keeps clauses matching configurable signal words
- Writes CSV: doc_id, clause_id, clause_text

Usage:
  python tos_clause_extractor.py --input input.html --doc-id meta_ai_tos_2024
  python tos_clause_extractor.py --input tos_dir
"""

import argparse, re, html, csv
from pathlib import Path
from typing import List

DEFAULT_SIGNALS = [
    r"\bmay not\b", r"\bmust\b", r"\bshall\b", r"\bwill\b", r"\bmay\b",
    r"\bare responsible\b", r"\bliable\b", r"\bagree\b", r"\bprohibited\b",
    r"\brequir(?:e|ed)\b", r"\bnot responsible\b", r"\breserve the right\b",
    r"\bsuspend\b", r"\bterminate\b", r"\bdelete\b", r"\bconsent\b",
    r"\bnotify\b", r"\baccount\b", r"\bprivacy\b", r"\bincluding\b",
    r"\bprovide\b", r"\buse\b", r"\bshare\b", r"\bcollect\b"
]

VERB_HINTS = [
    r"\bmay\b", r"\bmust\b", r"\bshall\b", r"\bwill\b", r"\bcan\b", r"\bcannot\b",
    r"\bare\b", r"\bis\b", r"\bwas\b", r"\bwere\b", r"\bbe\b", r"\bbeing\b", r"\bbeen\b",
    r"\bagree\b", r"\bagrees\b", r"\bagreed\b",
    r"\bconsent\b", r"\bprohibited\b", r"\brequired\b", r"\brequire\b",
    r"\bliable\b", r"\bresponsible\b", r"\bnotify\b", r"\bcollect\b", r"\bshare\b",
    r"\bsuspend\b", r"\bterminate\b", r"\bincluding\b", r"\bprovide\b"
]

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(errors="ignore")

def pre_normalize_html(text: str) -> str:
    """Normalize HTML with better list handling."""
    # Mark list items specially so we can handle them
    text = re.sub(r'(?i)<\s*li\s*>', '\n§LIST_ITEM§ ', text)
    text = re.sub(r'(?i)</\s*li\s*>', '\n', text)
    
    # Treat block-like tags as line breaks
    text = re.sub(r'(?i)</\s*(p|div|section|ul|ol)\s*>', '\n\n', text)
    text = re.sub(r'(?i)<\s*br\s*/?\s*>', '\n', text)
    
    # Bold tags can be significant (headers) so preserve them with markers
    text = re.sub(r'(?i)<\s*b\s*>', ' §BOLD§', text)
    text = re.sub(r'(?i)</\s*b\s*>', '§/BOLD§ ', text)
    
    # Remove remaining tags
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    
    # Insert missing spaces before a Capital letter following a lowercase letter
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    
    # Normalize whitespace but preserve paragraph breaks
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    return text.strip()

def extract_clauses_from_list(text: str, context: str = "") -> List[str]:
    """Extract clauses from text containing list items, preserving context."""
    clauses = []
    lines = text.split('\n')
    
    current_context = context
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this is a list item
        if '§LIST_ITEM§' in line:
            # Extract the list item text
            item_text = line.replace('§LIST_ITEM§', '').strip(' -•\u2022\t')
            if item_text and current_context:
                # Combine context with list item
                full_clause = f"{current_context} {item_text}"
                clauses.append(full_clause)
        else:
            # This might be context for following list items or a standalone clause
            # Remove bold markers
            clean_line = line.replace('§BOLD§', '').replace('§/BOLD§', '').strip()
            
            # Check if this looks like a heading/context (ends with colon or "would")
            if clean_line.endswith((':','would')) or re.search(r'(?:may not|must|shall|will|cannot)[\s:]*$', clean_line):
                current_context = clean_line.rstrip(':').strip()
            else:
                # Standalone clause
                if clean_line:
                    clauses.append(clean_line)
                current_context = ""
    
    return clauses

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences on major punctuation."""
    # Split on sentence punctuation followed by space and capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Also split on semicolons that separate independent clauses
    result = []
    for sent in sentences:
        # Split on semicolon followed by capital (likely independent clause)
        parts = re.split(r';\s*(?=[A-Z])', sent)
        result.extend([p.strip() for p in parts if p.strip()])
    
    return result

def rough_blocks(text: str) -> List[str]:
    """Split text into rough blocks for processing."""
    # Split on paragraph breaks
    blocks = text.split('\n\n')
    return [b.strip() for b in blocks if b.strip()]

def split_into_clauses(chunk: str) -> List[str]:
    """Split a chunk into individual clauses."""
    # First check if this contains list items
    if '§LIST_ITEM§' in chunk:
        # Extract context (text before first list item)
        parts = chunk.split('§LIST_ITEM§', 1)
        context = parts[0].strip()
        context = context.replace('§BOLD§', '').replace('§/BOLD§', '').strip()
        
        return extract_clauses_from_list(chunk, context)
    
    # No list items, treat as regular text
    chunk = chunk.replace('§BOLD§', '').replace('§/BOLD§', '')
    
    # Split into sentences
    clauses = split_into_sentences(chunk)
    
    return [c.strip(' -•\u2022;:') for c in clauses if c.strip()]

def strip_quotes(text: str) -> str:
    """Strip leading and trailing quotation marks from text."""
    return text.strip('"\'""''`')

def clean_clause_text(text: str) -> str:
    """Clean clause text by removing quotes, commas, and trailing section numbers."""
    text = strip_quotes(text)
    # Remove commas to prevent CSV formatting issues
    text = text.replace(',', '')
    # Remove trailing section numbers like "\n 1." or "\n 10."
    text = re.sub(r'\s*\n\s*\d+\.\s*$', '', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def is_heading_like(s: str) -> bool:
    """Check if text looks like a heading rather than a real clause."""
    if len(s) < 25:  # Increased minimum length
        return True
    
    # Check for standalone section numbers (e.g., "1.", "4.", "10.")
    if re.match(r'^\d+\.\s*$', s):
        return True
    
    # Mostly uppercase letters
    letters = re.sub(r'[^A-Za-z]', '', s)
    if letters and sum(1 for ch in letters if ch.isupper()) / max(1, len(letters)) > 0.7:
        return True
    
    # No verb hints at all -> likely a heading
    verb_re = re.compile("|".join(VERB_HINTS), flags=re.IGNORECASE)
    if not verb_re.search(s):
        return True
    
    return False

def build_regex(tokens: List[str]) -> re.Pattern:
    return re.compile("|".join(tokens), flags=re.IGNORECASE)

def iter_files(path: Path):
    if path.is_file():
        return [path]
    files = []
    for ext in ("*.txt", "*.html", "*.htm"):
        files.extend(path.rglob(ext))
    return files

def derive_doc_id(p: Path) -> str:
    s = re.sub(r'[^a-zA-Z0-9_]+', '_', p.stem).strip('_').lower()
    return s or "document"

def generate_output_path(input_path: Path) -> Path:
    """Generate output path with _filtered.csv suffix in filtered tos documents folder."""
    script_dir = Path(__file__).parent
    filtered_dir = script_dir / "filtered tos documents"
    filtered_dir.mkdir(exist_ok=True)
    
    if input_path.is_file():
        return filtered_dir / f"{input_path.stem}_filtered.csv"
    else:
        # For directory processing, create a single combined output file
        return filtered_dir / "all_tos_filtered.csv"

def main():
    ap = argparse.ArgumentParser(description="Improved extractor for ToS clauses.")
    ap.add_argument("--input", required=True, help="Input file or directory (.txt/.html)")
    ap.add_argument("--output", default=None, help="Output CSV path (defaults to filtered tos documents/all_tos_filtered.csv)")
    ap.add_argument("--doc-id", default=None, help="Doc ID for single-file mode; defaults to filename stem")
    ap.add_argument("--signals", default=None, help="Comma-separated regex terms for signal words")
    ap.add_argument("--minlen", type=int, default=40, help="Minimum clause length (characters)")
    ap.add_argument("--maxlen", type=int, default=800, help="Maximum clause length (characters)")
    ap.add_argument("--keep-headings", action="store_true", help="Do not drop heading-like strings")
    args = ap.parse_args()

    in_path = Path(args.input)
    
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = generate_output_path(in_path)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)

    signal_tokens = DEFAULT_SIGNALS if not args.signals else [t.strip() for t in args.signals.split(",") if t.strip()]
    sig_re = build_regex(signal_tokens)

    files = iter_files(in_path)
    if not files:
        raise SystemExit(f"No files found for {in_path}")

    rows = []
    for f in files:
        raw = read_text(f)
        norm = pre_normalize_html(raw)
        blocks = rough_blocks(norm)
        # Always derive doc_id from filename (unless explicitly provided for single file)
        doc_id = args.doc_id if (args.doc_id and len(files) == 1) else derive_doc_id(f)

        buf = []
        for blk in blocks:
            clauses = split_into_clauses(blk)
            for c in clauses:
                # Optional filters
                if not args.keep_headings and is_heading_like(c):
                    continue
                if not (args.minlen <= len(c) <= args.maxlen):
                    continue
                if not sig_re.search(c):
                    continue
                buf.append(c)

        # Deduplicate while preserving order
        seen = set()
        kept = []
        for c in buf:
            k = c.strip()
            if k not in seen:
                seen.add(k)
                kept.append(k)

        # Add clauses from this document to the combined rows
        for i, clause in enumerate(kept, start=1):
            cleaned_clause = clean_clause_text(clause)
            rows.append({"doc_id": doc_id, "clause_id": f"{i:03d}", "clause_text": cleaned_clause})

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["doc_id", "clause_id", "clause_text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} clauses from {len(files)} document(s) to {out_path}")

if __name__ == "__main__":
    main()