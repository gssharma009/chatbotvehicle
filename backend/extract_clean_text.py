import fitz
import re
from pathlib import Path

HEADER_FOOTER_TRASH = [
    r"electric vehicle information",
    r"characteristics of ev",
    r"warning",
    r"cautions",
    r"jsw mg",
    r"maintenance",
    r"table of contents",
    r"general information",
    r"page \d+",
]


def is_trash(line):
    l = line.lower()
    return any(re.search(p, l) for p in HEADER_FOOTER_TRASH)


def fix_broken_words(line):
    # Common OCR fixes (generic, not manual-specific)
    line = re.sub(r"(\w)-\s+(\w)", r"\1\2", line)  # self- contained → self-contained
    line = re.sub(r"\b([A-Za-z])\s+([A-Za-z])\b", r"\1\2", line)  # spac ed → spaced
    line = re.sub(r"\b([A-Za-z]{1,2})\s+([A-Za-z]{3,})", r"\1 \2", line)
    line = line.replace(" gure", " figure")
    line = line.replace(" a c ", " AC ")
    return line


def clean_line(line):
    # Remove symbols
    line = re.sub(r'[^A-Za-z0-9\s\.\,\;\:\(\)\-\?]', ' ', line)

    # Collapse spaces
    line = re.sub(r'\s+', ' ', line)

    # OCR fixes
    line = fix_broken_words(line)

    # Normalize common errors (generic)
    subs = {
        " off ": " of ",
        " u ": " you ",
        " hv ": " HV ",
        " ev ": " EV ",
        " ac ": " AC ",
    }
    for a, b in subs.items():
        line = line.replace(a, b)

    return line.strip()


def join_into_paragraphs(lines, max_len=350):
    """Turns many small lines into clean paragraphs for embeddings."""
    paragraphs = []
    buf = ""

    for line in lines:
        if not line:
            continue

        if len(buf) + len(line) + 1 < max_len:
            buf += " " + line
        else:
            paragraphs.append(buf.strip())
            buf = line

    if buf:
        paragraphs.append(buf.strip())

    return paragraphs


# ------------------------
# MAIN
# ------------------------
output_lines = []

for pdf_path in Path("docs").glob("*.pdf"):
    print(f"Extracting from {pdf_path.name}...")
    doc = fitz.open(pdf_path)

    for page in doc:
        text = page.get_text("text")

        for raw in text.split("\n"):
            line = raw.strip()
            if len(line) < 8:
                continue
            if is_trash(line):
                continue
            cleaned = clean_line(line)
            if len(cleaned) > 10:
                output_lines.append(cleaned)

paragraphs = join_into_paragraphs(output_lines)

with open("cleaned_manual_clean.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(paragraphs))

print(f"DONE → {len(paragraphs)} clean paragraphs.")
print("Now run create_embeddings.py")
