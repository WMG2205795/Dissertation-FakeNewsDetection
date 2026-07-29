import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List



def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_units_with_stats(units):
    """
    Clean url2text units while preserving order.
    Remove empty units, units containing encoding/PDF artefacts, and exact duplicates.
    Return both cleaned units and cleaning statistics.
    """
    if units is None:
        units = []

    if not isinstance(units, list):
        units = [str(units)]

    cleaned = []
    seen = set()

    stats = {
        "n_input_units": len(units),
        "removed_empty_units": 0,
        "removed_garbled_units": 0,
        "removed_duplicate_units": 0,
    }

    for unit in units:
        unit_clean = normalize_text(unit)

        # Remove empty units.
        if not unit_clean:
            stats["removed_empty_units"] += 1
            continue

        # Aggressive filter:
        # remove any unit containing Unicode replacement character.
        # This usually indicates PDF/OCR/encoding corruption.
        if "�" in unit_clean:
            stats["removed_garbled_units"] += 1
            continue

        # Remove exact duplicates after normalization.
        key = unit_clean.lower()
        if key in seen:
            stats["removed_duplicate_units"] += 1
            continue

        seen.add(key)
        cleaned.append(unit_clean)

    stats["n_cleaned_units"] = len(cleaned)

    return cleaned, stats


def load_json_or_jsonl(input_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load AVeriTeC knowledge-store files.

    Supports:
    1. Single JSON object
    2. JSON list
    3. Standard JSONL
    4. Multiple JSON objects concatenated together, including multi-line objects
    """
    input_path = Path(input_path)

    with input_path.open("r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # Case 1 / 2: normal JSON object or JSON list
    try:
        data = json.loads(text)

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]

        raise ValueError(f"Unsupported JSON root type: {type(data)}")

    except json.JSONDecodeError:
        pass

    # Case 3 / 4: continuously decode JSON objects from the full text
    decoder = json.JSONDecoder()
    entries = []
    idx = 0
    n = len(text)

    while idx < n:
        # Skip whitespace/newlines between JSON objects
        while idx < n and text[idx].isspace():
            idx += 1

        if idx >= n:
            break

        try:
            obj, next_idx = decoder.raw_decode(text, idx)
        except json.JSONDecodeError as e:
            context_start = max(0, idx - 300)
            context_end = min(n, idx + 800)
            context = text[context_start:context_end]

            raise ValueError(
                f"Cannot parse JSON object near character {idx} in {input_path}.\n"
                f"Original error: {e}\n\n"
                f"Context around error:\n{context}"
            ) from e

        if not isinstance(obj, dict):
            raise ValueError(
                f"Expected JSON object at character {idx}, but got {type(obj)}"
            )

        entries.append(obj)
        idx = next_idx

    return entries


def reconstruct_one_entry(
    entry: Dict[str, Any],
    doc_index: int = 0,
    file_stem: str = "unknown"
) -> Dict[str, Any]:
    """
    Convert one raw AVeriTeC URL-level record into one cleaned document-level record.
    The main output only keeps cleaned text to avoid storing garbled PDF/OCR artefacts.
    """
    claim_id = str(entry.get("claim_id", "unknown"))
    source_type = str(entry.get("type", "unknown"))
    query = entry.get("query", "")
    url = entry.get("url", "")

    raw_units = entry.get("url2text", [])
    if raw_units is None:
        raw_units = []
    if not isinstance(raw_units, list):
        raw_units = [str(raw_units)]

    cleaned, cleaning_stats = clean_units_with_stats(raw_units)

    doc_id = f"{file_stem}_{claim_id}_{source_type}_{doc_index:06d}"
    cleaned_text = " ".join(cleaned)

    return {
        "claim_id": claim_id,
        "doc_id": doc_id,
        "source_type": source_type,
        "query": query,
        "url": url,
        "original_file": file_stem,
        "original_index": doc_index,

        # Cleaned version for later chunking.
        "cleaned_units": cleaned,
        "cleaned_text": cleaned_text,

        # Quality flags and statistics.
        "is_empty": len(cleaned) == 0,
        "n_raw_units": len(raw_units),
        "n_cleaned_units": len(cleaned),
        "n_cleaned_chars": len(cleaned_text),

        # Cleaning statistics.
        "removed_empty_units": cleaning_stats["removed_empty_units"],
        "removed_garbled_units": cleaning_stats["removed_garbled_units"],
        "removed_duplicate_units": cleaning_stats["removed_duplicate_units"],
    }


def reconstruct_file(input_path: str | Path, output_path: str | Path) -> List[Dict[str, Any]]:
    """
    Reconstruct one raw JSON/JSONL file into document-level JSONL.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    entries = load_json_or_jsonl(input_path)

    documents = [
        reconstruct_one_entry(entry, i)
        for i, entry in enumerate(entries)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    total_removed_empty = sum(doc["removed_empty_units"] for doc in documents)
    total_removed_garbled = sum(doc["removed_garbled_units"] for doc in documents)
    total_removed_duplicate = sum(doc["removed_duplicate_units"] for doc in documents)

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Raw entries: {len(entries)}")
    print(f"Documents written: {len(documents)}")
    print(f"Empty documents: {sum(doc['is_empty'] for doc in documents)}")
    print(f"Non-empty documents: {sum(not doc['is_empty'] for doc in documents)}")
    print(f"Removed empty units: {total_removed_empty}")
    print(f"Removed garbled units: {total_removed_garbled}")
    print(f"Removed duplicate units: {total_removed_duplicate}")
    return documents

def iter_input_files(input_dir: str | Path, recursive: bool = True) -> List[Path]:
    input_dir = Path(input_dir)

    if recursive:
        files = list(input_dir.rglob("*.json")) + list(input_dir.rglob("*.jsonl"))
    else:
        files = list(input_dir.glob("*.json")) + list(input_dir.glob("*.jsonl"))

    return sorted(files)

def reconstruct_folder(
    input_dir: str | Path,
    output_path: str | Path,
    recursive: bool = True,
    drop_empty: bool = True
) -> None:
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    input_files = iter_input_files(input_dir, recursive=recursive)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_entries = 0
    total_written = 0
    total_empty_docs = 0
    total_removed_empty_units = 0
    total_removed_garbled_units = 0
    total_removed_duplicate_units = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for input_file in input_files:
            print(f"Processing: {input_file}")

            entries = load_json_or_jsonl(input_file)
            total_files += 1
            total_entries += len(entries)

            file_stem = input_file.stem

            for i, entry in enumerate(entries):
                doc = reconstruct_one_entry(
                    entry,
                    doc_index=i,
                    file_stem=file_stem
                )

                total_removed_empty_units += doc["removed_empty_units"]
                total_removed_garbled_units += doc["removed_garbled_units"]
                total_removed_duplicate_units += doc["removed_duplicate_units"]

                if doc["is_empty"]:
                    total_empty_docs += 1
                    if drop_empty:
                        continue

                out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                total_written += 1

    print("\nDone.")
    print(f"Input folder: {input_dir}")
    print(f"Output file: {output_path}")
    print(f"Files processed: {total_files}")
    print(f"Raw entries: {total_entries}")
    print(f"Documents written: {total_written}")
    print(f"Empty documents: {total_empty_docs}")
    print(f"Removed empty units: {total_removed_empty_units}")
    print(f"Removed garbled units: {total_removed_garbled_units}")
    print(f"Removed duplicate units: {total_removed_duplicate_units}")


if __name__ == "__main__":
    reconstruct_folder(
        input_dir=r"F:\train_combination",
        output_path=r"F:\documents.jsonl",
        recursive=True,
        drop_empty=True
    )