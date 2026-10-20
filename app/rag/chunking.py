"""Text chunking module with semantic boundary preservation and metadata attribution."""

from typing import List, Dict, Any, Optional


def recursive_character_split(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """Splits text recursively on natural semantic separators (paragraphs, lines, sentences).

    Ensures chunks do not exceed chunk_size while maintaining meaningful overlap.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the highest priority separator present in the text
    chosen_separator = ""
    for sep in separators:
        if sep == "" or sep in text:
            chosen_separator = sep
            break

    if chosen_separator:
        splits = text.split(chosen_separator)
    else:
        splits = list(text)

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for split in splits:
        split_len = len(split) + len(chosen_separator)
        if current_length + split_len > chunk_size and current_chunk:
            combined = chosen_separator.join(current_chunk).strip()
            if combined:
                chunks.append(combined)

            # Keep overlap by preserving tail elements from previous chunk
            overlap_acc: List[str] = []
            overlap_len = 0
            for item in reversed(current_chunk):
                item_len = len(item) + len(chosen_separator)
                if overlap_len + item_len <= chunk_overlap:
                    overlap_acc.insert(0, item)
                    overlap_len += item_len
                else:
                    break

            current_chunk = overlap_acc
            current_length = overlap_len

        current_chunk.append(split)
        current_length += split_len

    if current_chunk:
        remaining = chosen_separator.join(current_chunk).strip()
        if remaining:
            chunks.append(remaining)

    return chunks


def chunk_document(
    document: Dict[str, Any],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """Splits a single document into attributed chunks with metadata.

    Returns list of dicts with:
      - 'id': unique chunk identifier (e.g., 'type_2_diabetes.md_0')
      - 'text': chunk text content
      - 'source': source filename
      - 'title': document title
      - 'chunk_index': sequence integer
      - 'char_count': text length
    """
    raw_text = document.get("text", "")
    source = document.get("source", "unknown_source")
    title = document.get("title", source)

    raw_chunks = recursive_character_split(
        text=raw_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    attributed_chunks: List[Dict[str, Any]] = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunk_id = f"{source}_{idx}"
        attributed_chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "source": source,
            "title": title,
            "chunk_index": idx,
            "char_count": len(chunk_text),
        })

    return attributed_chunks


def chunk_all_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """Chunks a list of documents and flattens into a single list of attributed chunks."""
    all_chunks: List[Dict[str, Any]] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return all_chunks
