import json
from pathlib import Path
import sqlite3
import re
import numpy as np
import torch
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

"""
Config zone
"""

CLAIM_PATH = Path(r"E:\2026MainFiles\WMG_AAI 2025-2026\Dissertation\Project Code\AVeriTeC\data\internal_split\dev_claims_200.json")
CLAIM_ID_PATH = Path(r"E:\2026MainFiles\WMG_AAI 2025-2026\Dissertation\Project Code\AVeriTeC\data\internal_split\dev_ids_200.json")

CHUNK_CONFIG = {
    "sentence": {
        "db_path": Path(r"F:\internal_split\dev_sentence.db"),
        "embedding_path": Path(r"F:\internal_split\sentence_embedding"),
        "output_path": Path(r"E:\2026MainFiles\WMG_AAI 2025-2026\Dissertation\Project Code\chunking_test\report\embedding\sentence_report.json"),
        "table_name": "sentences",
        "id_column": "sentence_id",
        "text_column": "contents",
        "chunking_method": "sentence",
    },
    "word": {
        "db_path": Path(r"F:\internal_split\dev_chunks_200_overlap_50.db"),
        "embedding_path": Path(r"F:\internal_split\chunks_200_overlap_50_embedding"),
        "output_path": Path(r"E:\2026MainFiles\WMG_AAI 2025-2026\Dissertation\Project Code\chunking_test\report\embedding\chunks_200_overlap_50_report.json"),
        "table_name": "chunks",
        "id_column": "chunk_id",
        "text_column": "contents",
        "chunking_method": "word_overlap",
    },
}

CHUNK_MODE= "word"

if CHUNK_MODE not in CHUNK_CONFIG:
    raise ValueError(
        f"Unsupported CHUNK_MODE: {CHUNK_MODE}. "
        f"Choose from {list(CHUNK_CONFIG)}"
    )

CONFIG = CHUNK_CONFIG[CHUNK_MODE]

EMBEDDING_DIR = CONFIG["embedding_path"]
DB_PATH = CONFIG["db_path"]
OUTPUT_PATH = CONFIG["output_path"]
TABLE_NAME = CONFIG["table_name"]
ID_COLUMN = CONFIG["id_column"]
TEXT_COLUMN = CONFIG["text_column"]
CHUNKING_METHOD = CONFIG["chunking_method"]

LLM_MODEL_NAME = "qwen2.5:7b"
TEMPERATURE = 0.0



EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RETRIEVE_K=100
FINAL_TOP_K = 10
VECTORS_NORMALIZED = True
DEVICE = "cuda"


#LLM Calling

ALLOWED_LABELS = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]
llm = ChatOllama(
    model=LLM_MODEL_NAME,
    temperature=TEMPERATURE,
    top_p=1.0,
    seed=42
)

def call_llm(prompt):
    response = llm.invoke(prompt)
    if not response.content:
        raise ValueError("The LLM returned an empty response.")
    return str(response.content)

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

def normalize_contents(text):
    if text is None:
        return ""

    return " ".join(
        str(text).lower().split()
    )

def deduplicate_chunks(
    retrieved_chunks,
    final_k,
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

    unique_chunks.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    unique_count = len(unique_chunks)

    unique_chunks = unique_chunks[:final_k]

    for rank, item in enumerate(
        unique_chunks,
        start=1,
    ):
        item["rank"] = rank

    return unique_chunks, unique_count

#Read all claims and form one metrix

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
    (num_queries, RETRIEVE_K),
    -float("inf"),
    device=DEVICE,
)

best_shards = torch.full(
    (num_queries, RETRIEVE_K),
    -1,
    dtype=torch.long,
    device=DEVICE,
)

best_local_ids = torch.full(
    (num_queries, RETRIEVE_K),
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
        k=min(RETRIEVE_K, shard_vectors.shape[0]),
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
        k=RETRIEVE_K,
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
    for rank in range(RETRIEVE_K):
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

    for rank in range(RETRIEVE_K):
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

dense_topk_records = []
DENSE_TOPK_PATH = OUTPUT_PATH.with_name(
    OUTPUT_PATH.stem + "_topk.json"
)
for stored_claim_id, stored_candidates in dense_topk.items():
    dense_topk_records.append(
        {
            "claim_id": stored_claim_id,
            "retrieved_keys": stored_candidates,
        }
    )

save_json(
    DENSE_TOPK_PATH,
    dense_topk_records,
)

print(f"Dense top-k saved to: {DENSE_TOPK_PATH}")

connection = sqlite3.connect(
    f"file:{DB_PATH}?mode=ro",
    uri=True,
)

results = []

try:
    for index, (claim_id, item) in enumerate(
        zip(claim_ids, claim_records),
        start=1,
    ):
        
        claim_id = int(claim_id)
        claim = item["claim"]
        gold_label = item["label"]

        retrieved_chunks = []
        current_retrieved_keys = []
        raw_response = None
        unique_count = 0

        try:
            current_retrieved_keys = dense_topk[claim_id]

            retrieved_chunks = lookup_retrieved_chunks(
                connection=connection,
                retrieved_keys=current_retrieved_keys,
            )

            retrieved_chunks, unique_count = deduplicate_chunks(
                retrieved_chunks=retrieved_chunks,
                final_k=FINAL_TOP_K,
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
                "retriever": "Dense",
                "chunking_method": CHUNKING_METHOD,
                "embedding_model": EMBEDDING_MODEL,
                "model_name": LLM_MODEL_NAME,
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
                "retriever": "Dense",
                "chunking_method": CHUNKING_METHOD,
                "embedding_model": EMBEDDING_MODEL,
                "model_name": LLM_MODEL_NAME,
                "temperature": TEMPERATURE,
                "retrieved_evidence": retrieved_chunks,
                "raw_response": raw_response,
                "error": repr(exc),
            }

        results.append(
            output_record
        )

        save_json(
            OUTPUT_PATH,
            results,
        )

        print(
            f"[{index}/{len(claim_records)}] "
            f"Claim {claim_id}: "
            f"{output_record['predicted_label'] or 'ERROR'}"
        )

finally:
    connection.close()

print(f"Saved to: {OUTPUT_PATH}")