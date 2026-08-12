import json
from pathlib import Path

import sqlite3
import os
os.environ.setdefault("OPENAI_API_KEY", "not-used-for-bm25")
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pyserini.search.lucene import LuceneSearcher

"""
Config zone
"""
BASE_DIR = Path(__file__).resolve().parent.parent

CLAIM_PATH = BASE_DIR / "AVeriTeC" / "data" / "internal_split" / "dev_claims_200.json"
CLAIM_ID_PATH = BASE_DIR / "AVeriTeC" / "data" / "internal_split" / "dev_ids_200.json"



CHUNK_MODE = "sentence"  # Options: "sentence", "word_100_25", "word_200_50"
DEDUPLICATE = True


CHUNK_CONFIG = {
    "sentence": {
        "db_path": Path(r"F:\internal_split\dev_sentence.db"),
        "embedding_path": Path(
            r"F:\internal_split\sentence_embedding"
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
        "embedding_path": Path(
            r"F:\internal_split\chunks_100_overlap_25_embedding"
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
        "embedding_path": Path(
            r"F:\internal_split\chunks_200_overlap_50_embedding"
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

EMBEDDING_DIR = CONFIG["embedding_path"]
BM25_INDEX_PATH = CONFIG["bm25_index_path"]

DB_PATH = CONFIG["db_path"]
TABLE_NAME = CONFIG["table_name"]
ID_COLUMN = CONFIG["id_column"]
TEXT_COLUMN = CONFIG["text_column"]
CHUNKING_METHOD = CONFIG["chunking_method"]


CANDIDATE_K = 50

DENSE_RETRIEVE_K = 1000
BM25_RETRIEVE_K = 1000

RRF_K = 60

BM25_K1 = 1.2
BM25_B = 0.75

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTORS_NORMALIZED = True
DEVICE = "cuda"

OUTPUT_PATH = Path(
            BASE_DIR / "chunking_test"/"report" / "Hybrid" / f"{CHUNK_MODE}_Hybrid_retrieval_cache_{CANDIDATE_K}_Dedup_{DEDUPLICATE}.json"
        )


def save_json(
    output_path,
    records,
):
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





with CLAIM_PATH.open(encoding="utf-8") as f:
    claim_records = json.load(f)

with CLAIM_ID_PATH.open(encoding="utf-8") as f:
    claim_ids = json.load(f)
  
        
if len(claim_records) != len(claim_ids):
    raise ValueError(
        f"Claims/IDs mismatch: "
        f"{len(claim_records)} vs {len(claim_ids)}"
    )
        

claims = [item["claim"] for item in claim_records]

model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

# BGE official suggestion
query_texts = [
    "Represent this sentence for searching relevant passages: "
    + claim
    for claim in claims
]

query_vectors = model.encode(
    query_texts,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).to(torch.float32)

num_queries = len(claims)

best_scores = torch.full(
    (num_queries, DENSE_RETRIEVE_K),
    -float("inf"),
    device=DEVICE,
)

best_shards = torch.full(
    (num_queries, DENSE_RETRIEVE_K),
    -1,
    dtype=torch.long,
    device=DEVICE,
)

best_local_ids = torch.full(
    (num_queries, DENSE_RETRIEVE_K),
    -1,
    dtype=torch.long,
    device=DEVICE,
)

vector_files = sorted(
    EMBEDDING_DIR.glob("vectors_*.npy")
)

for shard_id, vector_file in enumerate(vector_files):
    array = np.load(
        vector_file,
        mmap_mode="r",
    )

    shard_vectors = torch.tensor(
        array,
        dtype=torch.float32,
        device=DEVICE,
    )

    if not VECTORS_NORMALIZED:
        shard_vectors = torch.nn.functional.normalize(
            shard_vectors,
            p=2,
            dim=1,
        )
    scores = query_vectors @ shard_vectors.T

    local_scores, local_ids = torch.topk(
        scores,
        k=min(DENSE_RETRIEVE_K, shard_vectors.shape[0]),
        dim=1,
    )

    shard_ids = torch.full_like(
        local_ids,
        shard_id,
    )

    combined_scores = torch.cat(
        [best_scores, local_scores],
        dim=1,
    )

    combined_shards = torch.cat(
        [best_shards, shard_ids],
        dim=1,
    )

    combined_local_ids = torch.cat(
        [best_local_ids, local_ids],
        dim=1,
    )

    best_scores, positions = torch.topk(
        combined_scores,
        k=DENSE_RETRIEVE_K,
        dim=1,
    )

    best_shards = torch.gather(
        combined_shards,
        1,
        positions,
    )

    best_local_ids = torch.gather(
        combined_local_ids,
        1,
        positions,
    )

    print(
        f"[{shard_id + 1}/{len(vector_files)}] "
        f"{vector_file.name}"
    )

    del array, shard_vectors, scores

best_scores = best_scores.cpu().float().numpy()
best_shards = best_shards.cpu().numpy()
best_local_ids = best_local_ids.cpu().numpy()

needed = {}

for query_index in range(num_queries):
    for rank in range(DENSE_RETRIEVE_K):
        shard_id = int(best_shards[query_index, rank])
        local_id = int(best_local_ids[query_index, rank])

        needed.setdefault(
            shard_id,
            set(),
        ).add(local_id)


resolved_keys = {}

for shard_id, local_ids in needed.items():
    vector_file = vector_files[shard_id]

    key_file = vector_file.with_name(
        vector_file.name.replace(
            "vectors_",
            "keys_",
        )
    ).with_suffix(".jsonl")

    with key_file.open(
        encoding="utf-8",
    ) as f:
        for local_id, line in enumerate(f):
            if local_id in local_ids:
                resolved_keys[
                    (shard_id, local_id)
                ] = json.loads(line)

dense_topk = {}

for query_index, claim_id in enumerate(claim_ids):
    retrieved_keys = []

    for rank in range(DENSE_RETRIEVE_K):
        shard_id = int(
            best_shards[query_index, rank]
        )

        local_id = int(
            best_local_ids[query_index, rank]
        )

        score = float(
            best_scores[query_index, rank]
        )

        key = resolved_keys[
            (shard_id, local_id)
        ]

        store_claim_id = key["claim_id"]
        record_id = int(key["record_id"])

        if "sentence_id" in key:
            chunk_id = int(key["sentence_id"])
        else:
            chunk_id = int(key["chunk_id"])

        retrieved_keys.append(
            {
                "rank": rank + 1,
                "score": score,
                "store_claim_id": store_claim_id,
                "record_id": record_id,
                "chunk_id": chunk_id,
            }
        )

    dense_topk[int(claim_id)] = retrieved_keys

del model
del query_vectors
torch.cuda.empty_cache()



"""
Above is the dense retrieval part, below is the BM25 retrieval part
"""

def parse_docid(docid: str) -> tuple[str, int, int]:
    store_id, record_id, sentence_id = docid.rsplit("_", 2)

    return (
        store_id,
        int(record_id),
        int(sentence_id),
    )

def retrieve_bm25_keys(
    searcher: LuceneSearcher,
    claim: str,
    top_k: int,
) -> list[dict]:

    hits = searcher.search(
        claim,
        k=top_k,
    )

    retrieved_keys = []

    for rank, hit in enumerate(
        hits,
        start=1,
    ):
        (
            store_claim_id,
            record_id,
            chunk_id,
        ) = parse_docid(hit.docid)

        retrieved_keys.append(
            {
                "rank": rank,
                "score": float(hit.score),
                "store_claim_id": store_claim_id,
                "record_id": record_id,
                "chunk_id": chunk_id,
            }
        )

    return retrieved_keys

searcher = LuceneSearcher(
    BM25_INDEX_PATH
)

searcher.set_bm25(
    k1=BM25_K1,
    b=BM25_B,
)

bm25_topk = {}

for index, (claim_id, claim) in enumerate(
    zip(claim_ids, claims),
    start=1,
):
    claim_id = int(claim_id)

    retrieved_keys = retrieve_bm25_keys(
        searcher=searcher,
        claim=claim,
        top_k=BM25_RETRIEVE_K,
    )

    bm25_topk[claim_id] = retrieved_keys

    print(
        f"[BM25 {index}/{len(claims)}] "
        f"Claim {claim_id}: "
        f"{len(retrieved_keys)} results"
    )

"""
Then read the dense and BM25 results, combine them using RRF, and save the final top-k results.
But before that, need to deduplicate the chunks based on their contents, so that don't have duplicate chunks in the final results.
"""

def lookup_retrieved_chunks(
    connection,
    retrieved_keys,
):
    retrieved_chunks = []

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

    for item in retrieved_keys:
        result = connection.execute(
            query,
            (
                item["store_claim_id"],
                item["record_id"],
                item["chunk_id"],
            ),
        ).fetchone()

        if result is None:
            contents = None
            source_url = None
            source_type = None
        else:
            contents, source_url, source_type = result

        docid = (
            f"{item['store_claim_id']}_"
            f"{item['record_id']}_"
            f"{item['chunk_id']}"
        )

        retrieved_chunks.append(
            {
                "rank": item["rank"],
                "docid": docid,
                "score": item["score"],
                "store_claim_id": item["store_claim_id"],
                "record_id": item["record_id"],
                "chunk_id": item["chunk_id"],
                "contents": contents,
                "source_url": source_url,
                "source_type": source_type,
            }
        )

    return retrieved_chunks

def normalize_contents(text):
    if text is None:
        return ""

    return " ".join(
        str(text).lower().split()
    )


def deduplicate_chunks(
    retrieved_chunks,
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

    for rank, item in enumerate(
        unique_chunks,
        start=1,
    ):
        item["rank"] = rank

    return unique_chunks

def apply_deduplication(
    retrieved_chunks,
    deduplicate,
):
    if deduplicate:
        return deduplicate_chunks(
            retrieved_chunks
        )

    return retrieved_chunks


connection = sqlite3.connect(
    f"file:{DB_PATH}?mode=ro",
    uri=True,
)

dense_deduplicated = {}
bm25_deduplicated = {}

try:
    for index, claim_id in enumerate(
        claim_ids,
        start=1,
    ):
        claim_id = int(claim_id)

        dense_keys = dense_topk[claim_id]
        bm25_keys = bm25_topk[claim_id]

        dense_chunks = lookup_retrieved_chunks(
            connection=connection,
            retrieved_keys=dense_keys,
        )

        bm25_chunks = lookup_retrieved_chunks(
            connection=connection,
            retrieved_keys=bm25_keys,
        )

        dense_candidates = apply_deduplication(
            retrieved_chunks=dense_chunks,
            deduplicate=DEDUPLICATE,
        )

        bm25_candidates = apply_deduplication(
            retrieved_chunks=bm25_chunks,
            deduplicate=DEDUPLICATE,
        )

        dense_deduplicated[claim_id] = (
            dense_candidates
        )

        bm25_deduplicated[claim_id] = (
            bm25_candidates
        )

        print(
            f"[Candidate {index}/{len(claim_ids)}] "
            f"Claim {claim_id}: "
            f"Dense {len(dense_chunks)}"
            f" -> {len(dense_candidates)}, "
            f"BM25 {len(bm25_chunks)}"
            f" -> {len(bm25_candidates)} "
            f"(dedup={DEDUPLICATE})"
        )

finally:
    connection.close()


"""
Finally, RRF
"""

def RRF_fusion(
    bm25_results,
    dense_results,
    rrf_k=60,
    candidate_k=50,
):
    fused = {}

    for item in bm25_results:
        key = normalize_contents(
            item.get("contents")
        )

        if not key:
            continue

        if key not in fused:
            fused[key] = item.copy()
            fused[key]["bm25_rank"] = None
            fused[key]["bm25_score"] = None
            fused[key]["dense_rank"] = None
            fused[key]["dense_score"] = None
            fused[key]["rrf_score"] = 0.0

        if fused[key]["bm25_rank"] is None:
            fused[key]["bm25_rank"] = item["rank"]
            fused[key]["bm25_score"] = item["score"]

            fused[key]["rrf_score"] += (
                1.0 / (rrf_k + item["rank"])
            )

    for item in dense_results:
        key = normalize_contents(
            item.get("contents")
        )

        if not key:
            continue

        if key not in fused:
            fused[key] = item.copy()
            fused[key]["bm25_rank"] = None
            fused[key]["bm25_score"] = None
            fused[key]["dense_rank"] = None
            fused[key]["dense_score"] = None
            fused[key]["rrf_score"] = 0.0

        if fused[key]["dense_rank"] is None:
            fused[key]["dense_rank"] = item["rank"]
            fused[key]["dense_score"] = item["score"]

            fused[key]["rrf_score"] += (
                1.0 / (rrf_k + item["rank"])
            )

    fused_results = list(
        fused.values()
    )

    fused_results.sort(
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    candidate_results = fused_results[
        :candidate_k
    ]

    for rank, item in enumerate(
        candidate_results,
        start=1,
    ):
        item["rank"] = rank

    return candidate_results


hybrid_results = []

for claim_id, claim_record in zip(
    claim_ids,
    claim_records,
):
    claim_id = int(claim_id)

    hybrid_candidates = RRF_fusion(
        bm25_results=bm25_deduplicated[claim_id],
        dense_results=dense_deduplicated[claim_id],
        rrf_k=RRF_K,
        candidate_k=CANDIDATE_K,
    )

    hybrid_results.append(
        {
            "claim_id": claim_id,
            "claim": claim_record["claim"],
            "gold_label": claim_record["label"],
            "retriever": "Hybrid_RRF",
            "chunking_method": CHUNKING_METHOD,

            "dense_retrieve_k": DENSE_RETRIEVE_K,
            "bm25_retrieve_k": BM25_RETRIEVE_K,

            "candidate_k": CANDIDATE_K,
            "deduplicated": DEDUPLICATE,

            "rrf_k": RRF_K,

            "retrieved_evidence": hybrid_candidates,
        }
    )


save_json(
    OUTPUT_PATH,
    hybrid_results,
)

print(
    f"Saved to: {OUTPUT_PATH}"
)