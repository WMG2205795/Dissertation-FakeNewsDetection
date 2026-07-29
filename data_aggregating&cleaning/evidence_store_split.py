import json
from pathlib import Path


EVIDENCE_PATH = Path(
    r"F:aggregated_raw.jsonl"
)

DEV_IDS_PATH = Path(
    r"E:\2026MainFiles\WMG_AAI 2025-2026"
    r"\Dissertation\Project Code\AVeriTeC\data"
    r"\internal_split\dev_ids_200.json"
)

TEST_IDS_PATH = Path(
    r"E:\2026MainFiles\WMG_AAI 2025-2026"
    r"\Dissertation\Project Code\AVeriTeC\data"
    r"\internal_split\test_ids_600.json"
)

OUTPUT_DIR = Path(
    r"F:\internal_split"
)


with DEV_IDS_PATH.open("r", encoding="utf-8") as file:
    dev_ids = set(json.load(file))

with TEST_IDS_PATH.open("r", encoding="utf-8") as file:
    test_ids = set(json.load(file))


overlap = dev_ids & test_ids

if overlap:
    raise RuntimeError(
        f"Data leakage detected: "
        f"{len(overlap)} IDs appear in both dev and test."
    )

print(f"Dev IDs: {len(dev_ids)}")
print(f"Test IDs: {len(test_ids)}")
print(f"Dev/Test overlap: {len(overlap)}")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEV_OUTPUT_PATH = OUTPUT_DIR / "dev_evidence_200.jsonl"
TEST_OUTPUT_PATH = OUTPUT_DIR / "test_evidence_600.jsonl"

total_lines = 0
invalid_json_lines = 0
missing_id_lines = 0

dev_record_count = 0
test_record_count = 0

found_dev_ids = set()
found_test_ids = set()

with (
    EVIDENCE_PATH.open("r", encoding="utf-8") as input_file,
    DEV_OUTPUT_PATH.open("w", encoding="utf-8") as dev_file,
    TEST_OUTPUT_PATH.open("w", encoding="utf-8") as test_file,
):
    for line_number, line in enumerate(input_file, start=1):
        total_lines += 1


        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            invalid_json_lines += 1

            print(
                f"Invalid JSON at line {line_number}: {error}"
            )
            continue

       
        raw_id = record.get("id")

        if raw_id is None:
            raw_id = record.get("claim_id")

        if raw_id is None:
            missing_id_lines += 1
            continue

        try:
            evidence_id = int(raw_id)
        except (TypeError, ValueError):
            missing_id_lines += 1
            continue

        if evidence_id in dev_ids:
            dev_file.write(line)
            if not line.endswith("\n"):
                dev_file.write("\n")

            dev_record_count += 1
            found_dev_ids.add(evidence_id)

        elif evidence_id in test_ids:
            test_file.write(line)

            if not line.endswith("\n"):
                test_file.write("\n")

            test_record_count += 1
            found_test_ids.add(evidence_id)

        if total_lines % 10_000 == 0:
            print(
                f"Processed: {total_lines:,} lines | "
                f"Dev records: {dev_record_count:,} | "
                f"Test records: {test_record_count:,}"
            )


missing_dev_ids = dev_ids - found_dev_ids
missing_test_ids = test_ids - found_test_ids


print("\n" + "=" * 70)
print("EVIDENCE EXTRACTION COMPLETED")
print("=" * 70)

print(f"Total evidence lines processed: {total_lines:,}")
print(f"Dev evidence records: {dev_record_count:,}")
print(f"Test evidence records: {test_record_count:,}")

print(f"\nExpected dev IDs: {len(dev_ids)}")
print(f"Found dev IDs: {len(found_dev_ids)}")
print(f"Missing dev IDs: {len(missing_dev_ids)}")

print(f"\nExpected test IDs: {len(test_ids)}")
print(f"Found test IDs: {len(found_test_ids)}")
print(f"Missing test IDs: {len(missing_test_ids)}")

print(f"\nInvalid JSON lines: {invalid_json_lines:,}")
print(f"Missing/invalid ID lines: {missing_id_lines:,}")

print(f"\nDev output: {DEV_OUTPUT_PATH}")
print(f"Test output: {TEST_OUTPUT_PATH}")


if missing_dev_ids:
    print(
        "\nMissing dev ID examples:",
        sorted(missing_dev_ids)[:20],
    )

if missing_test_ids:
    print(
        "\nMissing test ID examples:",
        sorted(missing_test_ids)[:20],
    )