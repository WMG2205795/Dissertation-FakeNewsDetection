import json
import re
import argparse
from pathlib import Path
import csv

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def is_control_char(ch: str) -> bool:
    return ord(ch) < 32 and ch not in "\n\r\t"


def inspect_text(text: str):
    has_non_ascii = not text.isascii()
    non_ascii_count = sum(1 for ch in text if not ch.isascii()) if has_non_ascii else 0
    has_replacement = "�" in text
    has_control = any(ord(ch) < 32 and ch not in "\n\r\t" for ch in text)
    has_cjk = CJK_RE.search(text) is not None

    return {
        "has_non_ascii": has_non_ascii,
        "non_ascii_count": non_ascii_count,
        "has_replacement": has_replacement,
        "has_control": has_control,
        "has_cjk": has_cjk,
    }


def is_suspicious(stats, text_len, non_ascii_ratio_threshold=0.02):
    if text_len == 0:
        return False
    if stats["has_replacement"]:
        return True
    if stats["has_control"]:
        return True
    if stats["has_cjk"]:
        return True
    if stats["non_ascii_count"] / text_len > non_ascii_ratio_threshold:
        return True
    return False


def preview(text, max_len=500):
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]

"""
Due to the large size of the aggregated JSONL file, 
I will implement a checkpointing mechanism to allow resuming the overview process from where it left off. 
The checkpoint will store the last processed byte offset, along with the current statistics and examples collected so far. 
This way, if the process is interrupted, we can load the checkpoint and continue processing without starting over.
Finally it will be saved in a summary CSV file for easy viewing and analysis.
"""

def save_checkpoint(checkpoint_path, input_path, byte_offset, stats, examples):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "input_path": str(input_path),
        "byte_offset": byte_offset,
        "stats": stats,
        "examples": examples,
    }

    with checkpoint_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def load_checkpoint(checkpoint_path, input_path):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        return None

    with checkpoint_path.open("r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    if checkpoint.get("input_path") != str(input_path):
        print("Checkpoint exists, but input_path is different. Starting from scratch.")
        return None

    return checkpoint


def save_summary_csv(summary_path, stats):
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for key, value in stats.items():
            writer.writerow([key, value])

        if stats["total_units"] > 0:
            writer.writerow([
                "suspicious_unit_ratio",
                stats["suspicious_units"] / stats["total_units"]
            ])



def overview_jsonl(
        input_path,
        max_examples=2,
        checkpoint_path="data_aggregating&cleaning/overview_checkpoint.json",
        summary_path="data_aggregating&cleaning/overview_summary.csv",
        save_every_lines=100000,
    ):
    input_path = Path(input_path)

    checkpoint = load_checkpoint(checkpoint_path, input_path)

    if checkpoint is not None:
        byte_offset = checkpoint["byte_offset"]
        stats = checkpoint["stats"]
        examples = checkpoint["examples"]

        print(f"Resuming from byte offset: {byte_offset:,}")
        print(f"Already processed lines: {stats['total_lines']:,}")
    else:
        byte_offset = 0

        stats = {
            "total_lines": 0,
            "empty_lines": 0,
            "parse_errors": 0,

            "total_records": 0,
            "missing_url2text": 0,
            "empty_url2text": 0,

            "total_units": 0,
            "empty_units": 0,
            "suspicious_units": 0,
            "records_with_suspicious_units": 0,

            "units_with_non_ascii": 0,
            "units_with_replacement": 0,
            "units_with_control": 0,
            "units_with_cjk": 0,
        }

        examples = {
            "empty_url2text_examples": [],
            "suspicious_examples": [],
            "parse_error_examples": [],
        }

        print("No checkpoint found. Starting from scratch.")

    with input_path.open("rb") as f:
        f.seek(byte_offset)

        for line_bytes in f:
            byte_offset += len(line_bytes)
            stats["total_lines"] += 1

            line = line_bytes.decode("utf-8", errors="replace")

            if stats["total_lines"] % save_every_lines == 0:
                save_checkpoint(checkpoint_path, input_path, byte_offset, stats, examples)
                save_summary_csv(summary_path, stats)
                print(
                    f"Processed {stats['total_lines']:,} lines, "
                    f"byte offset {byte_offset:,}. Checkpoint saved."
                )

            if not line.strip():
                stats["empty_lines"] += 1
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                stats["parse_errors"] += 1

                if len(examples["parse_error_examples"]) < max_examples:
                    examples["parse_error_examples"].append({
                        "line_no": stats["total_lines"],
                        "error": repr(e),
                        "line_preview": preview(line),
                    })

                continue

            stats["total_records"] += 1

            units = obj.get("url2text", None)

            if units is None:
                stats["missing_url2text"] += 1

                if len(examples["empty_url2text_examples"]) < max_examples:
                    examples["empty_url2text_examples"].append({
                        "line_no": stats["total_lines"],
                        "reason": "missing_url2text",
                        "claim_id": obj.get("claim_id"),
                        "type": obj.get("type"),
                        "url": obj.get("url"),
                    })

                continue

            if not isinstance(units, list):
                units = [str(units)]

            if len(units) == 0:
                stats["empty_url2text"] += 1

                if len(examples["empty_url2text_examples"]) < max_examples:
                    examples["empty_url2text_examples"].append({
                        "line_no": stats["total_lines"],
                        "reason": "empty_url2text_list",
                        "claim_id": obj.get("claim_id"),
                        "type": obj.get("type"),
                        "url": obj.get("url"),
                    })

                continue

            record_has_suspicious = False
            non_empty_unit_count = 0

            for unit_index, unit in enumerate(units):
                text = str(unit)
                text_len = len(text)

                stats["total_units"] += 1

                if not text.strip():
                    stats["empty_units"] += 1
                    continue

                non_empty_unit_count += 1

                char_stats = inspect_text(text)

                stats["units_with_non_ascii"] += char_stats["has_non_ascii"]
                stats["units_with_replacement"] += char_stats["has_replacement"]
                stats["units_with_control"] += char_stats["has_control"]
                stats["units_with_cjk"] += char_stats["has_cjk"]

                if is_suspicious(char_stats, text_len):
                    stats["suspicious_units"] += 1
                    record_has_suspicious = True

                    if len(examples["suspicious_examples"]) < max_examples:
                        examples["suspicious_examples"].append({
                            "line_no": stats["total_lines"],
                            "unit_index": unit_index,
                            "claim_id": obj.get("claim_id"),
                            "type": obj.get("type"),
                            "url": obj.get("url"),
                            "char_stats": char_stats,
                            "unit_preview": preview(text, 800),
                        })

            if non_empty_unit_count == 0:
                stats["empty_url2text"] += 1

                if len(examples["empty_url2text_examples"]) < max_examples:
                    examples["empty_url2text_examples"].append({
                        "line_no": stats["total_lines"],
                        "reason": "all_units_empty",
                        "claim_id": obj.get("claim_id"),
                        "type": obj.get("type"),
                        "url": obj.get("url"),
                    })

            if record_has_suspicious:
                stats["records_with_suspicious_units"] += 1

    save_checkpoint(checkpoint_path, input_path, byte_offset, stats, examples)
    save_summary_csv(summary_path, stats)

    print("\n" + "=" * 80)
    print("DOCUMENT OVERVIEW")
    print("=" * 80)

    print(f"Input file: {input_path}")
    print(f"Total lines: {stats['total_lines']:,}")
    print(f"Empty lines: {stats['empty_lines']:,}")
    print(f"JSON parse errors: {stats['parse_errors']:,}")
    print(f"Parsed records: {stats['total_records']:,}")

    print("\nURL2TEXT")
    print(f"Missing url2text records: {stats['missing_url2text']:,}")
    print(f"Empty url2text records: {stats['empty_url2text']:,}")
    print(f"Total url2text units: {stats['total_units']:,}")
    print(f"Empty units: {stats['empty_units']:,}")

    print("\nSUSPICIOUS CHARACTERS")
    print(f"Suspicious units: {stats['suspicious_units']:,}")
    print(f"Records with suspicious units: {stats['records_with_suspicious_units']:,}")
    print(f"Units with non-ASCII chars: {stats['units_with_non_ascii']:,}")
    print(f"Units with replacement chars: {stats['units_with_replacement']:,}")
    print(f"Units with control chars: {stats['units_with_control']:,}")
    print(f"Units with CJK chars: {stats['units_with_cjk']:,}")

    if stats['total_units'] > 0:
        print(f"Suspicious unit ratio: {stats['suspicious_units'] / stats['total_units']:.4%}")

    print("\nEMPTY URL2TEXT EXAMPLES")
    for ex in examples["empty_url2text_examples"]:
        print("-" * 80)
        print(json.dumps(ex, ensure_ascii=False, indent=2))

    print("\nSUSPICIOUS UNIT EXAMPLES")
    for ex in examples["suspicious_examples"]:
        print("-" * 80)
        print(json.dumps(ex, ensure_ascii=False, indent=2))

    print("\nJSON PARSE ERROR EXAMPLES")
    for ex in examples["parse_error_examples"]:
        print("-" * 80)
        print(json.dumps(ex, ensure_ascii=False, indent=2))

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to merged JSONL file."
    )

    parser.add_argument(
        "--max_examples",
        type=int,
        default=2,
        help="Number of examples to print for each issue type."
    )

    args = parser.parse_args()

    overview_jsonl(
        input_path=args.input_path,
        max_examples=args.max_examples
    )