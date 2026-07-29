import sqlite3
from pathlib import Path
import os
os.environ.setdefault("OPENAI_API_KEY", "not-used-for-bm25")

from pyserini.search.lucene import LuceneSearcher

import json
import re

from langchain_ollama import ChatOllama





CHUNK_MODE = "word"

CLAIM_PATH= Path(r"E:\2026MainFiles\WMG_AAI 2025-2026\Dissertation\Project Code\AVeriTeC\data\internal_split\dev_claims_200.json")

CLAIM_ID_PATH= Path(r"E:\2026MainFiles\WMG_AAI 2025-2026\Dissertation\Project Code\AVeriTeC\data\internal_split\dev_ids_200.json")

MODEL_NAME="qwen2.5:7b"
TEMPERATURE=0.0 #数值越低输出确定

CHUNK_CONFIG = {
    "sentence": {
        "index_path": r"F:\internal_split\BM25_Sentence_Indexing",
        "db_path": Path(r"F:\internal_split\dev_sentence.db"),
        "output_path": Path(r"E:\2026MainFiles\WMG_AAI 2025-2026\Dissertation\Project Code\chunking_test\report\sentence_report.json"),
        "table_name": "sentences",
        "id_column": "sentence_id",
        "text_column": "contents",
        "chunking_method": "sentence",
    },
    "word": {
        "index_path": r"F:\internal_split\BM25_chunks_400_overlap_100_Indexing",
        "db_path": Path(r"F:\internal_split\dev_chunks_400_overlap_100.db"),
        "output_path": Path(r"E:\2026MainFiles\WMG_AAI 2025-2026\Dissertation\Project Code\chunking_test\chunks_400_overlap_100_report.json"),
        "table_name": "chunks",
        "id_column": "chunk_id",
        "text_column": "contents",
        "chunking_method": "word_overlap",
    },
}

if CHUNK_MODE not in CHUNK_CONFIG:
    raise ValueError(
        f"Unsupported CHUNK_MODE: {CHUNK_MODE}. "
        f"Choose from {list(CHUNK_CONFIG)}"
    )

CONFIG = CHUNK_CONFIG[CHUNK_MODE]

INDEX_PATH = CONFIG["index_path"]
DB_PATH = CONFIG["db_path"]
OUTPUT_PATH = CONFIG["output_path"]
TABLE_NAME = CONFIG["table_name"]
ID_COLUMN = CONFIG["id_column"]
TEXT_COLUMN = CONFIG["text_column"]
CHUNKING_METHOD = CONFIG["chunking_method"]

ALLOWED_LABELS = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]

TOP_K = 10
BM25_K1 = 0.9
BM25_B = 0.4

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    top_p=1.0,
    seed=42
)

def call_llm(prompt):
    response = llm.invoke(prompt)
    if not response.content:
        raise ValueError("The LLM returned an empty response.")
    return str(response.content)


def parse_docid(docid: str) -> tuple[str, int, int]:
    store_id, record_id, sentence_id = docid.rsplit("_", 2)

    return (
        store_id,
        int(record_id),
        int(sentence_id),
    )


def lookup_chunks(
    connection: sqlite3.Connection,
    target_keys: set[tuple[str, int, int]],
) -> dict[tuple[str, int, int], tuple]:

    found = {}

    query = f"""
        SELECT
            {TEXT_COLUMN},
            source_url,
            source_type
        FROM {TABLE_NAME}
        WHERE claim_id = ?
          AND record_id = ?
          AND {ID_COLUMN} = ?
    """

    for key in target_keys:
        store_claim_id, record_id, chunk_id = key

        result = connection.execute(
            query,
            (
                store_claim_id,
                record_id,
                chunk_id,
            ),
        ).fetchone()

        if result is not None:
            found[key] = result

    return found


def retrieve_bm25(
    searcher: LuceneSearcher,
    connection: sqlite3.Connection,
    claim: str,
    top_k: int,
) -> list[dict]:

    hits = searcher.search(claim, k=top_k)

    parsed_hits = []

    for rank, hit in enumerate(hits, start=1):
        key = parse_docid(hit.docid)

        parsed_hits.append(
            {
                "rank": rank,
                "docid": hit.docid,
                "key": key,
                "score": float(hit.score),
            }
        )

    target_keys = {
        item["key"]
        for item in parsed_hits
    }

    chunk_map = lookup_chunks(
        connection=connection,
        target_keys=target_keys,
    )

    retrieved_chunks = []

    for item in parsed_hits:
        store_claim_id, record_id, chunk_id = item["key"]

        result = chunk_map.get(item["key"])

        if result is None:
            contents = None
            source_url = None
            source_type = None
        else:
            contents, source_url, source_type = result

        retrieved_chunks.append(
            {
                "rank": item["rank"],
                "docid": item["docid"],
                "score": item["score"],
                "store_claim_id": store_claim_id,
                "record_id": record_id,
                "chunk_id": chunk_id,
                "contents": contents,
                "source_url": source_url,
                "source_type": source_type,
            }
        )

    return retrieved_chunks


def format_evidence(retrieved_chunks: list[dict]) -> str:
    evidence_blocks = []

    for item in retrieved_chunks:
        contents = item.get("contents")

        if contents is None or not str(contents).strip():
            continue

        evidence_blocks.append(
            "\n".join(
                [
                    f"[Evidence {item['rank']}]",
                    f"Text: {contents}",
                    f"Source URL: {item.get('source_url') or 'Unavailable'}",
                    f"Source type: {item.get('source_type') or 'Unknown'}",
                ]
            )
        )

    if not evidence_blocks:
        return "[No usable evidence was retrieved.]"

    return "\n\n".join(evidence_blocks)


def build_prompt(
    claim: str,
    retrieved_chunks: list[dict],
) -> str:

    evidence_text = format_evidence(retrieved_chunks)

    return f"""
            You are evaluating a factual claim using retrieved evidence.

            CLAIM TO VERIFY:
            {claim}

            RETRIEVED EVIDENCE:
            {evidence_text}

            Assess the claim using only the retrieved evidence above.

            Choose exactly one label:

            - Supported: the evidence supports the central claim.
            - Refuted: the evidence contradicts the central claim.
            - Not Enough Evidence: the evidence is insufficient to support or refute the claim.
            - Conflicting Evidence/Cherrypicking: the evidence is meaningfully conflicting, or the claim presents evidence selectively in a misleading way.

            Return valid JSON only in this format:

            {{
            "predicted_label": "one of the four labels",
            "reason": "a concise explanation grounded in the retrieved evidence"
            }}
            """.strip()




def parse_llm_response(response_text: str) -> dict:
    text = response_text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

    result = json.loads(text)

    predicted_label = result.get("predicted_label")
    reason = result.get("reason")

    if predicted_label not in ALLOWED_LABELS:
        raise ValueError(
            f"Invalid predicted label: {predicted_label}"
        )

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Missing or empty reason.")

    return {
        "predicted_label": predicted_label,
        "reason": reason.strip(),
    }


def save_json(
    output_path: Path,
    records: list[dict],
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2,
        )

def load_completed_claim_ids(
    output_path: Path,
) -> set[int]:

    completed = set()

    if not output_path.exists():
        return completed

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            claim_id = record.get("claim_id")

            if claim_id is not None:
                completed.add(int(claim_id))

    return completed

def main():
    searcher = LuceneSearcher(INDEX_PATH)
    searcher.set_bm25(
        k1=BM25_K1,
        b=BM25_B,
    )

    connection = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True,
    )

    with CLAIM_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dev_claims = json.load(file)

    with CLAIM_ID_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        dev_ids = json.load(file)

    if len(dev_claims) != len(dev_ids):
        connection.close()

        raise ValueError(
            f"Claims/IDs length mismatch: "
            f"{len(dev_claims)} vs {len(dev_ids)}"
        )

    completed_claim_ids = load_completed_claim_ids(
        OUTPUT_PATH
    )

    results = []

    try:
        for index, (claim_id, item) in enumerate(
            zip(dev_ids, dev_claims),
            start=1,
        ):
            claim_id = int(claim_id)
            claim = item["claim"]
            gold_label = item["label"]

            if claim_id in completed_claim_ids:
                print(
                    f"[{index}/{len(dev_claims)}] "
                    f"Skipping claim {claim_id}"
                )
                continue

            retrieved_chunks = []
            raw_response = None

            try:
                retrieved_chunks = retrieve_bm25(
                    searcher=searcher,
                    connection=connection,
                    claim=claim,
                    top_k=TOP_K,
                )

                prompt = build_prompt(
                    claim=claim,
                    retrieved_chunks=retrieved_chunks,
                )

                raw_response = call_llm(
                    prompt
                )

                parsed_response = parse_llm_response(
                    raw_response
                )

                output_record = {
                    "claim_id": claim_id,
                    "claim": claim,
                    "gold_label": gold_label,
                    "predicted_label": (
                        parsed_response["predicted_label"]
                    ),
                    "reason": parsed_response["reason"],
                    "retriever": "BM25",
                    "chunking_method": CHUNKING_METHOD,
                    "top_k": TOP_K,
                    "bm25_k1": BM25_K1,
                    "bm25_b": BM25_B,
                    "model_name": MODEL_NAME,
                    "temperature": TEMPERATURE,
                    "retrieved_evidence": retrieved_chunks,
                    "raw_response": raw_response,
                    "error": None,
                }

            except Exception as exc:
                output_record = {
                    "claim_id": claim_id,
                    "claim": claim,
                    "gold_label": gold_label,
                    "predicted_label": None,
                    "reason": None,
                    "retriever": "BM25",
                    "chunking_method": CHUNKING_METHOD,
                    "top_k": TOP_K,
                    "bm25_k1": BM25_K1,
                    "bm25_b": BM25_B,
                    "model_name": MODEL_NAME,
                    "temperature": TEMPERATURE,
                    "retrieved_evidence": retrieved_chunks,
                    "raw_response": raw_response,
                    "error": repr(exc),
                }

            results.append(output_record)

            save_json(
                OUTPUT_PATH,
                results,
            )

            print(
                f"[{index}/{len(dev_claims)}] "
                f"Claim {claim_id}: "
                f"{output_record['predicted_label'] or 'ERROR'}"
            )

    finally:
        connection.close()

if __name__ == "__main__":
    main()

#