import json
import random
from collections import Counter, defaultdict
from pathlib import Path


INPUT_PATH = Path(
    r"E:\2026MainFiles\WMG_AAI 2025-2026\Dissertation\Project Code\AVeriTeC\data\train.json"
)

OUTPUT_DIR = Path(
    r"E:\2026MainFiles\WMG_AAI 2025-2026\Dissertation\Project Code\AVeriTeC\data\internal_split"
)

RANDOM_SEED = 42

DEV_COUNTS = {
    "Supported": 54,
    "Refuted": 114,
    "Conflicting Evidence/Cherrypicking": 12,
    "Not Enough Evidence": 20,
}

TEST_COUNTS = {
    "Supported": 162,
    "Refuted": 342,
    "Conflicting Evidence/Cherrypicking": 36,
    "Not Enough Evidence": 60,
}


def save_json(data, path):
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )



with INPUT_PATH.open("r", encoding="utf-8") as file:
    claims = json.load(file)

print(f"Total claims: {len(claims)}")


ids_by_label = defaultdict(list)

for original_id, item in enumerate(claims):
    label = item["label"]
    ids_by_label[label].append(original_id)


for label in DEV_COUNTS:
    required = DEV_COUNTS[label] + TEST_COUNTS[label]
    available = len(ids_by_label[label])

    if required > available:
        raise ValueError(
            f"Not enough samples for {label}: "
            f"required {required}, available {available}"
        )

rng = random.Random(RANDOM_SEED)

dev_ids = set()
test_ids = set()

for label in DEV_COUNTS:
    candidate_ids = ids_by_label[label].copy()
    rng.shuffle(candidate_ids)

    dev_count = DEV_COUNTS[label]
    test_count = TEST_COUNTS[label]
    # After shuffling, we can just slice the list to get the required number of IDs for dev and test sets.
    label_dev_ids = candidate_ids[:dev_count]

    label_test_ids = candidate_ids[
        dev_count:dev_count + test_count
    ]

    dev_ids.update(label_dev_ids)
    test_ids.update(label_test_ids)


overlap = dev_ids & test_ids

print("dataleakage items:", len(overlap))
print("DEV_num:", len(dev_ids))
print("TEST_num:", len(test_ids))

dev_claims = [
    item
    for original_id, item in enumerate(claims)
    if original_id in dev_ids
]

test_claims = [
    item
    for original_id, item in enumerate(claims)
    if original_id in test_ids
]

dev_distribution = Counter(
    item["label"] for item in dev_claims
)

test_distribution = Counter(
    item["label"] for item in test_claims
)

if dev_distribution != Counter(DEV_COUNTS):
    raise RuntimeError(
        f"Incorrect dev distribution:\n"
        f"Expected: {DEV_COUNTS}\n"
        f"Actual: {dict(dev_distribution)}"
    )

if test_distribution != Counter(TEST_COUNTS):
    raise RuntimeError(
        f"Incorrect test distribution:\n"
        f"Expected: {TEST_COUNTS}\n"
        f"Actual: {dict(test_distribution)}"
    )


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

save_json(
    dev_claims,
    OUTPUT_DIR / "dev_claims_200.json",
)

save_json(
    test_claims,
    OUTPUT_DIR / "test_claims_600.json",
)

save_json(
    sorted(dev_ids),
    OUTPUT_DIR / "dev_ids_200.json",
)

save_json(
    sorted(test_ids),
    OUTPUT_DIR / "test_ids_600.json",
)

print("\nDev distribution:")
for label, count in dev_distribution.items():
    print(f"{label}: {count}")

print("\nTest distribution:")
for label, count in test_distribution.items():
    print(f"{label}: {count}")

print(f"\nDev IDs: {len(dev_ids)}")
print(f"Test IDs: {len(test_ids)}")
print(f"Dev/Test overlap: {len(overlap)}")
print(f"Random seed: {RANDOM_SEED}")
print(f"Output directory: {OUTPUT_DIR}")