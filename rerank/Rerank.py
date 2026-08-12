import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

CHUNK_MODE = "sentence"  # Options: "sentence", "word_100_25", "word_200_50"
RETRIEVE_MODE = "Dense"  # Options: "BM25", "Dense", "Hybrid"
DEDUPLICATE = False  # Options: True, False
RERANK = False
FINAL_TOP_K = 10

INPUT_PATH = Path(
            BASE_DIR / "chunking_test"/"report" / f"{RETRIEVE_MODE}" / f"{CHUNK_MODE}_{RETRIEVE_MODE}_retrieval_cache_50_Dedup_{DEDUPLICATE}.json"
        )

OUTPUT_PATH = Path(
            BASE_DIR / "rerank"/"report" / f"{RETRIEVE_MODE}" / f"{CHUNK_MODE}_retrieval_top{FINAL_TOP_K}_rerank_{RERANK}_Dedup_{DEDUPLICATE}.json"
        )




def load_json(
    input_path: Path,
):
    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    output_path: Path,
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


def build_final_evidence(
    retrieved_evidence,
    rerank,
    final_top_k,
):
    if not rerank:
        return retrieved_evidence[:final_top_k]

    raise NotImplementedError(
        "Reranker has not been implemented yet."
    )


def main():

    retrieval_results = load_json(
        INPUT_PATH
    )

    final_results = []

    for item in retrieval_results:

        final_evidence = build_final_evidence(
            retrieved_evidence=item[
                "retrieved_evidence"
            ],
            rerank=RERANK,
            final_top_k=FINAL_TOP_K,
        )

        output_record = item.copy()

        output_record["retrieved_evidence"] = final_evidence

        output_record["reranked"] = RERANK

        output_record["top_k"] = FINAL_TOP_K

        final_results.append(output_record)

    save_json(
        OUTPUT_PATH,
        final_results,
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()