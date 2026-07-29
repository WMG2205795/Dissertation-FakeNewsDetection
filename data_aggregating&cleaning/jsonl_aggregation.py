"""
This file is used to merge the separate evidence store into one single JSONL file. 
It is used to reduce the number of files in the evidence store, which can be very large and slow to read.
The corresponding id is kept, which can be used to retrieve the original evidence store if needed. 
Also, the evidence store id is the same as the id of claims. 
Hence, when separate test set from original train set, the claim and corresponding evidence store can be selected by the same id.

"""


from pathlib import Path
import argparse
import os
import time
import shutil


def format_size(num_bytes: int) -> str:
    gb = num_bytes / (1024 ** 3)
    mb = num_bytes / (1024 ** 2)
    if gb >= 1:
        return f"{gb:.2f} GB"
    return f"{mb:.2f} MB"


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def find_jsonl_files(input_dir: Path, recursive: bool = True):
    """
    Find source files whose extension may be .json or .jsonl,
    but whose content is assumed to be JSONL: one JSON object per line.
    """
    if recursive:
        files = sorted(
            list(input_dir.rglob("*.json")) +
            list(input_dir.rglob("*.jsonl"))
        )
    else:
        files = sorted(
            list(input_dir.glob("*.json")) +
            list(input_dir.glob("*.jsonl"))
        )

    return files


def copy_one_file_binary(src_path: Path, out_f, buffer_size: int = 1024 * 1024 * 16) -> int:
    """
    Copy one JSONL file into an already opened output file.
    Return number of bytes copied.

    This does not parse JSON. It assumes each input file is valid JSONL.
    """
    bytes_copied = 0
    last_byte = None

    with src_path.open("rb") as in_f:
        while True:
            chunk = in_f.read(buffer_size)
            if not chunk:
                break

            out_f.write(chunk)
            bytes_copied += len(chunk)
            last_byte = chunk[-1]

    # Ensure boundary between two jsonl files.
    # If a file does not end with newline, add one.
    if bytes_copied > 0 and last_byte not in (10, 13):  # \n or \r
        out_f.write(b"\n")
        bytes_copied += 1

    return bytes_copied


def merge_jsonl_files(
    input_dir: str,
    output_path: str,
    recursive: bool = True,
    delete_after_merge: bool = True,
    buffer_size_mb: int = 16,
    fsync_each_file: bool = True,
):
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = find_jsonl_files(input_dir, recursive=recursive)

    # Avoid reading the output file if it is inside the input directory.
    files = [
        f for f in files
        if f.resolve() != output_path.resolve()
    ]

    if not files:
        print("No .jsonl files found.")
        return

    total_size = sum(f.stat().st_size for f in files)
    total_files = len(files)

    print("=" * 80)
    print("JSONL MERGE STARTED")
    print("=" * 80)
    print(f"Input directory : {input_dir}")
    print(f"Output file     : {output_path}")
    print(f"Files found     : {total_files}")
    print(f"Total size      : {format_size(total_size)}")
    print(f"Recursive       : {recursive}")
    print(f"Delete source   : {delete_after_merge}")
    print(f"fsync each file : {fsync_each_file}")
    print("=" * 80)

    start_time = time.time()
    processed_size = 0
    processed_files = 0
    failed_files = []

    buffer_size = buffer_size_mb * 1024 * 1024

    # Append mode: if output already exists, new content will be appended.
    # If you want a fresh merge, delete old output file before running.
    with output_path.open("ab") as out_f:
        for idx, src_path in enumerate(files, start=1):
            file_start = time.time()
            file_size = src_path.stat().st_size

            print()
            print(f"[{idx}/{total_files}] Processing: {src_path.name}")
            print(f"File size: {format_size(file_size)}")

            try:
                copied = copy_one_file_binary(
                    src_path=src_path,
                    out_f=out_f,
                    buffer_size=buffer_size,
                )

                out_f.flush()

                if fsync_each_file:
                    os.fsync(out_f.fileno())

                # Only delete after successful copy + flush/fsync.
                if delete_after_merge:
                    src_path.unlink()  # Permanent delete, not recycle bin.

                processed_files += 1
                processed_size += file_size

                elapsed = time.time() - start_time
                file_elapsed = time.time() - file_start

                speed = processed_size / elapsed if elapsed > 0 else 0
                remaining_size = max(total_size - processed_size, 0)
                eta = remaining_size / speed if speed > 0 else 0

                progress = processed_size / total_size * 100 if total_size > 0 else 0

                print(f"Copied bytes     : {format_size(copied)}")
                print(f"Deleted source   : {delete_after_merge}")
                print(f"File time        : {format_time(file_elapsed)}")
                print(f"Overall progress : {progress:.2f}%")
                print(f"Processed        : {format_size(processed_size)} / {format_size(total_size)}")
                print(f"Average speed    : {format_size(speed)}/s")
                print(f"Elapsed          : {format_time(elapsed)}")
                print(f"ETA              : {format_time(eta)}")

            except Exception as e:
                print(f"FAILED: {src_path}")
                print(f"Error : {repr(e)}")
                failed_files.append(str(src_path))
                continue

    total_elapsed = time.time() - start_time

    print()
    print("=" * 80)
    print("JSONL MERGE FINISHED")
    print("=" * 80)
    print(f"Files processed : {processed_files}/{total_files}")
    print(f"Size processed  : {format_size(processed_size)} / {format_size(total_size)}")
    print(f"Output file     : {output_path}")
    print(f"Total time      : {format_time(total_elapsed)}")

    if failed_files:
        error_log = output_path.parent / "merge_failed_files.txt"
        with error_log.open("w", encoding="utf-8") as f:
            for item in failed_files:
                f.write(item + "\n")

        print(f"Failed files    : {len(failed_files)}")
        print(f"Error log saved : {error_log}")
    else:
        print("Failed files    : 0")

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge many JSONL files into one JSONL file and permanently delete source files after successful merge."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing source .jsonl files."
    )

    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to merged output .jsonl file."
    )

    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Only search the top-level input directory, not subfolders."
    )

    parser.add_argument(
        "--no_delete",
        action="store_true",
        help="Do not delete source files after copying. Useful for testing."
    )

    parser.add_argument(
        "--buffer_size_mb",
        type=int,
        default=16,
        help="Copy buffer size in MB. Default: 16."
    )

    parser.add_argument(
        "--no_fsync",
        action="store_true",
        help="Do not fsync after each file. Faster but less safe if power is lost."
    )

    args = parser.parse_args()

    merge_jsonl_files(
        input_dir=args.input_dir,
        output_path=args.output_path,
        recursive=not args.no_recursive,
        delete_after_merge=not args.no_delete,
        buffer_size_mb=args.buffer_size_mb,
        fsync_each_file=not args.no_fsync,
    )