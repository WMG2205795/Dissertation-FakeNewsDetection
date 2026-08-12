import json
import re
from pathlib import Path

from langchain_ollama import ChatOllama


BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_NAME = "qwen2.5:7b" # qwen2.5:7b, qwen3:30b-a3b-instruct-2507-q4_K_M

SAFE_MODEL_NAME = MODEL_NAME.replace(":", "_").replace(".", "_")

RETRIEVAL_PATH = Path(
    BASE_DIR/"report"/"BM25"/"word_200_50_BM25_retrieval_cache.json"
)

OUTPUT_PATH = Path(
    BASE_DIR/f"report"/"BM25"/f"word_200_50_BM25_{SAFE_MODEL_NAME}.json"
)




TEMPERATURE = 0.0


ALLOWED_LABELS = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    top_p=1.0,
    seed=42,
)


def call_llm(prompt):

    response = llm.invoke(
        prompt
    )

    if not response.content:
        raise ValueError(
            "The LLM returned an empty response."
        )

    return str(
        response.content
    )


def format_evidence(
    retrieved_chunks: list[dict],
) -> str:

    evidence_blocks = []

    for item in retrieved_chunks:

        contents = item.get(
            "contents"
        )

        if (
            contents is None
            or not str(contents).strip()
        ):
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

        return (
            "[No usable evidence was retrieved.]"
        )


    return "\n\n".join(
        evidence_blocks
    )


def build_prompt(
    claim: str,
    retrieved_chunks: list[dict],
) -> str:

    evidence_text = format_evidence(
        retrieved_chunks
    )


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


def parse_llm_response(
    response_text: str,
) -> dict:

    text = response_text.strip()


    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()


    result = json.loads(
        text
    )


    predicted_label = result.get(
        "predicted_label"
    )

    reason = result.get(
        "reason"
    )


    if predicted_label not in ALLOWED_LABELS:

        raise ValueError(
            f"Invalid predicted label: "
            f"{predicted_label}"
        )


    if (
        not isinstance(reason, str)
        or not reason.strip()
    ):

        raise ValueError(
            "Missing or empty reason."
        )


    return {
        "predicted_label": predicted_label,
        "reason": reason.strip(),
    }


def save_json(
    output_path: Path,
    records: list[dict],
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

    with RETRIEVAL_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        retrieval_results = json.load(
            file
        )


    results = load_completed_results(
        OUTPUT_PATH
    )


    completed_claim_ids = {
        int(item["claim_id"])
        for item in results
    }


    for index, item in enumerate(
        retrieval_results,
        start=1,
    ):

        claim_id = int(
            item["claim_id"]
        )


        if claim_id in completed_claim_ids:

            print(
                f"[{index}/{len(retrieval_results)}] "
                f"Skipping claim {claim_id}"
            )

            continue


        claim = item["claim"]

        gold_label = item["gold_label"]

        retrieved_chunks = item[
            "retrieved_evidence"
        ]


        raw_response = None


        try:

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
                    parsed_response[
                        "predicted_label"
                    ]
                ),

                "reason": (
                    parsed_response[
                        "reason"
                    ]
                ),

                "retriever": item.get(
                    "retriever"
                ),

                "chunking_method": item.get(
                    "chunking_method"
                ),

                "top_k": item.get(
                    "top_k"
                ),

                "bm25_k1": item.get(
                    "bm25_k1"
                ),

                "bm25_b": item.get(
                    "bm25_b"
                ),

                "model_name": MODEL_NAME,

                "temperature": TEMPERATURE,

                "retrieved_evidence": (
                    retrieved_chunks
                ),

                "raw_response": (
                    raw_response
                ),

                "error": None,
            }


        except Exception as exc:

            output_record = {

                "claim_id": claim_id,

                "claim": claim,

                "gold_label": gold_label,

                "predicted_label": None,

                "reason": None,

                "retriever": item.get(
                    "retriever"
                ),

                "chunking_method": item.get(
                    "chunking_method"
                ),

                "top_k": item.get(
                    "top_k"
                ),

                "bm25_k1": item.get(
                    "bm25_k1"
                ),

                "bm25_b": item.get(
                    "bm25_b"
                ),

                "model_name": MODEL_NAME,

                "temperature": TEMPERATURE,

                "retrieved_evidence": (
                    retrieved_chunks
                ),

                "raw_response": (
                    raw_response
                ),

                "error": repr(
                    exc
                ),
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
            f"[{index}/{len(retrieval_results)}] "
            f"Claim {claim_id}: "
            f"{output_record['predicted_label'] or 'ERROR'}"
        )


if __name__ == "__main__":
    main()