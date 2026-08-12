import json
from pathlib import Path

from tqdm import tqdm

from src.prompt import PROMPT
from src.llm_loader import load_open_llm
from src.evidence import format_evidence

def load_retrieval_cache(path):
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON structure: {type(data)}")

    return data


def run_RAG(retrieval_cache_path, output_path, model_name, limit=None, temperature=0.0):
    retrieval_result = load_retrieval_cache(retrieval_cache_path)

    if limit is not None:
        retrieval_result = retrieval_result[:limit]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    structured_llm = load_open_llm(model_name=model_name, temperature=temperature)
    chain = PROMPT | structured_llm

    records = []

    for item in tqdm(retrieval_result, desc="Running RAG"):
        try:
            claim = item["claim"]
            retrieved_chunks = item["retrieved_evidence"]

            evidence_text = format_evidence(retrieved_chunks)

            result = chain.invoke({
                "claim": claim,
                "evidence": evidence_text
            })

            record = {
                "claim_id": item["claim_id"],
                "claim": claim,
                "gold_label": item["gold_label"],
                "predicted_label": result.label,
                "reason": result.reason,

                "retriever": item.get("retriever"),
                "chunking_method": item.get("chunking_method"),
                "retrieved_evidence": retrieved_chunks,

                "model_name": model_name,
                "temperature": temperature,
                "error": None,
            }

        except Exception as e:
            record = {
                "claim_id": item["claim_id"],
                "claim": claim,
                "gold_label": item["gold_label"],
                "predicted_label": result.label,
                "reason": result.reason,

                "retriever": item.get("retriever"),
                "chunking_method": item.get("chunking_method"),
                "retrieved_evidence": retrieved_chunks,

                "model_name": model_name,
                "temperature": temperature,
                "error": None,
            }


        records.append(record)

    with output_path.open("w", encoding="utf-8") as out:
        json.dump(records, out, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    CHUNK_MODE = "sentence"  # Options: "sentence", "word_100_25", "word_200_50"
    RETRIEVE_MODE = "Dense"  # Options: "BM25", "Dense", "Hybrid"
    DEDUPLICATE = False  # Options: True, False
    RERANK = False
    FINAL_TOP_K = 10

    retrieval_cache_path =Path(
            BASE_DIR /"rerank"/"report" / f"{RETRIEVE_MODE}" / f"{CHUNK_MODE}_retrieval_top{FINAL_TOP_K}_rerank_{RERANK}_Dedup_{DEDUPLICATE}.json"
    )

    
    model_name="qwen2.5:7b"
    output_path=f"output/{RETRIEVE_MODE} / {CHUNK_MODE}_retrieval_top{FINAL_TOP_K}_rerank_{RERANK}_Dedup_{DEDUPLICATE}_result.json"
    limit=None
    temperature = 0.0  

    run_RAG(retrieval_cache_path, output_path, model_name, limit, temperature)  