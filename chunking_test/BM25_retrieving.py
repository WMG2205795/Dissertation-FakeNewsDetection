import sqlite3
from pathlib import Path
import json

import os
os.environ.setdefault("OPENAI_API_KEY", "not-used-for-bm25")
from pyserini.search.lucene import LuceneSearcher

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CLAIM_PATH = BASE_DIR / "AVeriTeC" / "data" / "internal_split" / "dev_claims_200.json"
CLAIM_ID_PATH = BASE_DIR / "AVeriTeC" / "data" / "internal_split" / "dev_ids_200.json"

CHUNK_MODE = "word_200_50"
# Options:
# "sentence"
# "word_100_25"
# "word_200_50"

RETRIEVE_K = 1000
CANDIDATE_K = 50

DEDUPLICATE = False


CHUNK_CONFIG = {

    "sentence": {
        "db_path": Path(
            r"F:\internal_split\dev_sentence.db"
        ),
        "bm25_index_path": (
            r"F:\internal_split\BM25_Sentence_Indexing"
        ),
        "table_name": "sentences",
        "id_column": "sentence_id",
        "chunking_method": "sentence",
        "text_column": "contents",
    },

    "word_100_25": {
        "db_path": Path(
            r"F:\internal_split\dev_chunks_100_overlap_25.db"
        ),
        "bm25_index_path": (
            r"F:\internal_split\BM25_chunks_100_overlap_25_Indexing"
        ),
        "table_name": "chunks",
        "id_column": "chunk_id",
        "chunking_method": "word_100_overlap_25",
        "text_column": "contents",
    },

    "word_200_50": {
        "db_path": Path(
            r"F:\internal_split\dev_chunks_200_overlap_50.db"
        ),
        "bm25_index_path": (
            r"F:\internal_split\BM25_chunks_200_overlap_50_Indexing"
        ),
        "table_name": "chunks",
        "id_column": "chunk_id",
        "chunking_method": "word_200_overlap_50",
        "text_column": "contents",
    },
}


if CHUNK_MODE not in CHUNK_CONFIG:
    raise ValueError(
        f"Unsupported CHUNK_MODE: {CHUNK_MODE}. "
        f"Choose from {list(CHUNK_CONFIG)}"
    )


CONFIG = CHUNK_CONFIG[CHUNK_MODE]

INDEX_PATH = CONFIG["bm25_index_path"]
DB_PATH = CONFIG["db_path"]
OUTPUT_PATH = Path(
            BASE_DIR / "chunking_test"/"report" / "BM25" / f"{CHUNK_MODE}_BM25_retrieval_cache_{CANDIDATE_K}_Dedup_{DEDUPLICATE}.json"
        )
TABLE_NAME = CONFIG["table_name"]
ID_COLUMN = CONFIG["id_column"]
TEXT_COLUMN = CONFIG["text_column"]
CHUNKING_METHOD = CONFIG["chunking_method"]




BM25_K1 = 1.2
BM25_B = 0.75


def parse_docid(docid: str) -> tuple[str, int, int]:

    store_id, record_id, chunk_id = docid.rsplit("_", 2)

    return (
        store_id,
        int(record_id),
        int(chunk_id),
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

"""
Below is the function of deduplication
"""

def normalize_contents(text):
    if text is None:
        return ""

    return " ".join(
        str(text).lower().split()
    )

def deduplicate_chunks(
    retrieved_chunks,
    candidate_k,
):
    unique_chunks = []
    content_map = {}

    for item in retrieved_chunks:
        normalized = normalize_contents(
            item.get("contents")
        )

        if not normalized:
            continue

        provenance = {
            "store_claim_id": item["store_claim_id"],
            "record_id": item["record_id"],
            "chunk_id": item["chunk_id"],
            "source_url": item.get("source_url"),
            "source_type": item.get("source_type"),
        }

        if normalized in content_map:
            content_map[normalized][
                "duplicate_provenance"
            ].append(provenance)
            continue

        kept_item = item.copy()

        kept_item["duplicate_provenance"] = [
            provenance
        ]

        content_map[normalized] = kept_item
        unique_chunks.append(kept_item)

        if len(unique_chunks) >= candidate_k:
            break

    for rank, item in enumerate(
        unique_chunks,
        start=1,
    ):
        item["rank"] = rank

    return unique_chunks

"""
The switch of deduplication is implemented in the following function
"""

def build_candidate_pool(
    retrieved_chunks,
    deduplicate,
    candidate_k,
):
    if deduplicate:

        return deduplicate_chunks(
            retrieved_chunks=retrieved_chunks,
            candidate_k=candidate_k,
        )

    candidate_chunks = retrieved_chunks[
        :candidate_k
    ]

    for rank, item in enumerate(
        candidate_chunks,
        start=1,
    ):
        item["rank"] = rank

    return candidate_chunks

def retrieve_bm25(
    searcher: LuceneSearcher,
    connection: sqlite3.Connection,
    claim: str,
    top_k: int,
) -> list[dict]:

    hits = searcher.search(
        claim,
        k=top_k,
    )

    parsed_hits = []

    for rank, hit in enumerate(
        hits,
        start=1,
    ):

        key = parse_docid(
            hit.docid
        )

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

        result = chunk_map.get(
            item["key"]
        )

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


def load_completed_results(
    output_path: Path,
) -> list[dict]:

    if not output_path.exists():
        return []

    try:

        with output_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return []


def main():

    searcher = LuceneSearcher(
        INDEX_PATH
    )

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


    results = load_completed_results(
        OUTPUT_PATH
    )

    completed_claim_ids = {
        int(item["claim_id"])
        for item in results
    }


    try:

        for index, (claim_id, item) in enumerate(
            zip(dev_ids, dev_claims),
            start=1,
        ):

            claim_id = int(
                claim_id
            )

            claim = item["claim"]

            gold_label = item["label"]


            if claim_id in completed_claim_ids:

                print(
                    f"[{index}/{len(dev_claims)}] "
                    f"Skipping claim {claim_id}"
                )

                continue


            try:

                raw_retrieved_chunks = retrieve_bm25(
                    searcher=searcher,
                    connection=connection,
                    claim=claim,
                    top_k=RETRIEVE_K,
                )

                retrieved_chunks = build_candidate_pool(
                    retrieved_chunks=raw_retrieved_chunks,
                    deduplicate=DEDUPLICATE,
                    candidate_k=CANDIDATE_K,
                )

                output_record = {

                    "claim_id": claim_id,

                    "claim": claim,

                    "gold_label": gold_label,

                    "retriever": "BM25",

                    "chunking_method": CHUNKING_METHOD,

                    "retrieve_k": RETRIEVE_K,

                    "candidate_k": CANDIDATE_K,

                    "deduplicated": DEDUPLICATE,

                    "bm25_k1": BM25_K1,

                    "bm25_b": BM25_B,

                    "retrieved_evidence": retrieved_chunks,

                    "error": None,
                }


            except Exception as exc:

                output_record = {

                    "claim_id": claim_id,

                    "claim": claim,

                    "gold_label": gold_label,

                    "retriever": "BM25",

                    "chunking_method": CHUNKING_METHOD,

                    "retrieve_k": RETRIEVE_K,

                    "candidate_k": CANDIDATE_K,

                    "deduplicated": DEDUPLICATE,

                    "bm25_k1": BM25_K1,

                    "bm25_b": BM25_B,

                    "retrieved_evidence": [],

                    "error": repr(exc),
                }


            results.append(
                output_record
            )

            completed_claim_ids.add(
                claim_id
            )

            save_json(
                OUTPUT_PATH,
                results,
            )


            print(
                f"[{index}/{len(dev_claims)}] "
                f"Claim {claim_id}: "
                f"Retrieved "
                f"{len(output_record['retrieved_evidence'])} chunks"
            )


    finally:

        connection.close()


if __name__ == "__main__":
    main()