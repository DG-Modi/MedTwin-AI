"""Document loader and text cleaner for medical knowledge base."""

import re
from pathlib import Path
from typing import List, Dict, Any, Union


def clean_medical_text(text: str) -> str:
    """Normalizes and cleans raw document text.

    - Replaces excessive newlines and whitespace.
    - Preserves markdown headers and bullet points.
    - Strips non-printable characters.
    """
    if not text:
        return ""
    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing and leading whitespace
    return text.strip()


def extract_title(text: str, default_title: str) -> str:
    """Extracts the first markdown level 1 or 2 header as the document title."""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line.replace("# ", "").strip()
        elif line.startswith("## "):
            return line.replace("## ", "").strip()
    return default_title


def load_document(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a single markdown or text medical file, cleans it, and extracts metadata.

    Returns a dict with keys: 'text', 'source', 'title', 'path'.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    cleaned_text = clean_medical_text(raw_content)
    title = extract_title(cleaned_text, default_title=path.stem.replace("_", " ").title())

    return {
        "text": cleaned_text,
        "source": path.name,
        "title": title,
        "path": str(path),
    }


def load_all_documents(directory_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Recursively loads all markdown (.md) and text (.txt) files in the directory."""
    dir_path = Path(directory_path).resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        raise NotADirectoryError(f"Directory not found: {dir_path}")

    documents = []
    supported_extensions = {".md", ".txt"}

    for file_path in sorted(dir_path.rglob("*")):
        if file_path.suffix.lower() in supported_extensions and file_path.is_file():
            doc = load_document(file_path)
            if doc["text"]:
                documents.append(doc)

    return documents
