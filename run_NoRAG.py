import json
from pathlib import Path

from tqdm import tqdm

from src.prompt import PROMPT
from src.llm_loader import load_open_llm

def load_claims(path):
    path = Path(path)
    claims = []

    with path.open("r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        return claims
  
    data = json.loads(raw_text)

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON structure: {type(data)}")

    for idx, obj in enumerate(data):
        claim_id = (
            obj.get("claim_id")
            or obj.get("id")
            or f"claim_{idx:06d}"
        )

        claim = obj.get("claim") or obj.get("claim_text")
        label = obj.get("label") or obj.get("gold_label")

        if claim is None:
            print(f"Warning: missing claim at index {idx}")
            continue

        claims.append({
            "claim_id": claim_id,
            "claim": claim,
            "gold_label": label,
        })

    return claims


def run_NoRAG(input_path, output_path, model_name, limit=None, temperature=0.0):
    claims = load_claims(input_path)

    if limit is not None:
        claims = claims[:limit]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    structured_llm = load_open_llm(model_name=model_name, temperature=temperature)
    chain = PROMPT | structured_llm

    records = []

    for item in tqdm(claims, desc="Running No-RAG"):
        try:
            result = chain.invoke({"claim": item["claim"]})

            record = {
                "claim_id": item["claim_id"],
                "claim": item["claim"],
                "gold_label": item["gold_label"],
                "predicted_label": result.label,
                
                "reason": result.reason,
                "model_name": model_name,
                "error": None,
            }

        except Exception as e:
            record = {
                "claim_id": item["claim_id"],
                "claim": item["claim"],
                "gold_label": item["gold_label"],
                "predicted_label": None,
                
                "reason": None,
                "model_name": model_name,
                "error": repr(e),
            }

        records.append(record)

    with output_path.open("w", encoding="utf-8") as out:
        json.dump(records, out, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    input_path="AVeriTeC/data/train.json"
    model_name="qwen2.5:7b"
    if model_name == "qwen2.5:7b":
        output_path=f"output/results_NoRAG_Qwen.json"
    limit=None
    temperature = 0.0  

    run_NoRAG(input_path, output_path, model_name, limit, temperature)  