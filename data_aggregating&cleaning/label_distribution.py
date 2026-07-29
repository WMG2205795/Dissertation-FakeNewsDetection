import json
from collections import Counter
from pathlib import Path


input_path = Path(r"E:\2026MainFiles\WMG_AAI 2025-2026\Dissertation\Project Code\AVeriTeC\data\train.json")

with input_path.open("r", encoding="utf-8") as file:
    data = json.load(file)

label_counts = Counter(
    item["label"]
    for item in data
)

total = len(data)

print(f"Total claims: {total}")

for label, count in label_counts.items():
    percentage = count / total * 100

    print(
        f"{label}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )

