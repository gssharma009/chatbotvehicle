# repair_clean_text.py
# Usage: python repair_clean_text.py input_path output_path

import sys, re
from pathlib import Path

# small safe list of single-letter tokens to preserve
SAFE_SINGLE = {"a", "i", "A", "I"}

# common OCR word substitutions (extend as needed)
COMMON_SUBS = {
    r'\boff\b': 'of',
    r'\bu\b': 'you',
    r'\bse\b': 'use',
    r'\b0\b': '0',   # leave alone, just example
}

HEADER_FOOTER_TRASH = [
    r"electric vehicle information",
    r"characteristics of ev",
    r"warning",
    r"cautions",
    r"maintenance",
    r"table of contents",
    r"general information",
    r"page \d+",
    r"chapter \d+",
]

def is_trash(line):
    ll = line.lower()
    return any(re.search(p, ll) for p in HEADER_FOOTER_TRASH)

def join_spaced_letters(text):
    """
    Join patterns like:
      S a f e t y   -> Safety
      S a f e t y S y s t e m s -> SafetySystems (we then add a space between words later)
      Sa fe ty -> Safety
      in ood -> inood (may be wrong) — we further check and avoid joining very long joins blindly
    Heuristic:
      - Match groups of 2..10 fragments separated by spaces where fragments are 1-3 letters
      - Join them (remove inner spaces), but do not change if resulting string would be >25 chars (safety)
    """
    def repl(m):
        grp = m.group(0)
        joined = re.sub(r'\s+', '', grp)
        # safety: if join is huge, don't join
        if len(joined) > 25:
            return grp
        return joined

    # pattern: (1-3 letters) (space+1-3 letters){2,}
    out = re.sub(r'(?:\b[A-Za-z]{1,3}\b(?:\s+)){2,}[A-Za-z]{1,3}\b', repl, text)
    return out

def fix_hyphen_linebreaks(text):
    # remove hyphen + newline or hyphen + spaces that indicate line split
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    return text

def fix_internal_word_spaces(text):
    """
    For cases like 'a c c i d e n t' or 'Sa fe ty', join them.
    Also fix cases where two letter groups are split erroneously: 'in ood' -> 'in ood' (we don't join if first piece is short word).
    Heuristic: find sequences of tokens made of letters only where >2 tokens and average token length <=3 → join them.
    """
    def repl(m):
        tokens = m.group(0).split()
        avg = sum(len(t) for t in tokens) / len(tokens)
        # preserve if tokens contain known short words like 'in', 'to' at start followed by something (avoid joining)
        if avg <= 3 and len(tokens) >= 3:
            return "".join(tokens)
        return m.group(0)
    return re.sub(r'\b(?:[A-Za-z]{1,4}\s+){2,}[A-Za-z]{1,4}\b', repl, text)

def remove_weird_spacing(text):
    # remove spaces before punctuation, collapse multiple spaces
    text = re.sub(r'\s+([.,;:?!])', r'\1', text)
    text = re.sub(r'\s{2,}', ' ', text)
    # ensure a space after comma/period if missing
    text = re.sub(r'([.,;:?!])([A-Za-z0-9])', r'\1 \2', text)
    return text.strip()

def sentence_case_paragraph(p):
    # Very light sentence-casing: lowercase then capitalize first letter of sentence.
    p = p.strip()
    if not p:
        return p
    p = p[0].upper() + p[1:]
    return p

def apply_common_subs(line):
    for pat, repl in COMMON_SUBS.items():
        line = re.sub(pat, repl, line, flags=re.IGNORECASE)
    return line

def advanced_clean_line(line):
    if is_trash(line):
        return None

    # normalize newlines already removed; do lower-level fixes:
    line = fix_hyphen_linebreaks(line)
    line = join_spaced_letters(line)
    line = fix_internal_word_spaces(line)
    line = apply_common_subs(line)

    # Remove non-printable junk, allow basic punctuation
    line = re.sub(r'[^A-Za-z0-9\s\.\,\;\:\(\)\-\?\/\|]', ' ', line)

    line = remove_weird_spacing(line)

    # Re-run join for any leftovers
    line = join_spaced_letters(line)

    # Trim and skip too-short lines
    line = line.strip()
    if len(line) < 6:
        return None

    return line

def process_file(in_path, out_path):
    p = Path(in_path)
    if not p.exists():
        print("Input file not found:", in_path)
        return

    out_lines = []
    with open(p, 'r', encoding='utf-8') as f:
        raw_lines = f.readlines()

    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        cleaned = advanced_clean_line(raw)
        if cleaned:
            out_lines.append(cleaned)

    # Join small lines into paragraphs of ~300 chars
    paragraphs = []
    buf = ""
    for line in out_lines:
        if len(buf) + len(line) + 1 <= 320:
            if buf:
                buf += " " + line
            else:
                buf = line
        else:
            paragraphs.append(sentence_case_paragraph(buf))
            buf = line
    if buf:
        paragraphs.append(sentence_case_paragraph(buf))

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(paragraphs))

    print(f"Saved cleaned output: {out_path} ({len(paragraphs)} paragraphs)")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python repair_clean_text.py input.txt output.txt")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    process_file(in_path, out_path)
