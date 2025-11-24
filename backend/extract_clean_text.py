import fitz  # PyMuPDF
import re
from pathlib import Path
from bs4 import BeautifulSoup

# -------------------------------
# CLEANING UTILITIES
# -------------------------------

def strip_html(text: str) -> str:
    """Remove HTML if any accidentally appears in a PDF."""
    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)


def normalize_text(text: str) -> str:
    """Normalize spacing, remove noise, and unify formatting."""
    # Remove non-alphanumeric except safe punctuation
    text = re.sub(r'[^A-Za-z0-9\s\.\,\;\:\(\)\-\?]', ' ', text)

    # Fix multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def english_fixes(text: str) -> str:
    """Optional lightweight grammar/typo normalization."""
    replacements = {
        " off ": " of ",
        " u ": " you ",
        " nd ": " and ",
        " se ": " use ",
        " cant ": " can't ",
        " wont ": " won't ",
    }
    text = f" {text} "
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.strip()


def clean_line(line: str) -> str:
    """Full cleaning pipeline for each extracted line."""
    if not line or len(line.strip()) < 2:
        return ""

    line = strip_html(line)
    line = normalize_text(line)
    line = english_fixes(line)

    return line.strip()


# -------------------------------
# PDF EXTRACTION + PROCESSING
# -------------------------------

output = []
for pdf_path in Path("docs").glob("*.pdf"):
    print(f"Extracting clean text from {pdf_path.name}...")
    doc = fitz.open(pdf_path)

    for page in doc:
        text = page.get_text("text")
        raw_lines = text.split("\n")

        # Clean + filter
        for l in raw_lines:
            cl = clean_line(l)
            if len(cl) > 10:  # ignore tiny junk lines
                output.append(cl)

# -------------------------------
# REMOVE DUPLICATES WHILE PRESERVING ORDER
# -------------------------------
seen = set()
final_output = []
for line in output:
    if line not in seen:
        seen.add(line)
        final_output.append(line)

# -------------------------------
# SAVE OUTPUT
# -------------------------------
with open("manual_clean.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(final_output))

print(f"\nDONE! Clean manual saved: manual_clean.txt")
print(f"Total original lines: {len(output)}")
print(f"Total cleaned + unique lines: {len(final_output)}")
print("Ready for embeddings!")
