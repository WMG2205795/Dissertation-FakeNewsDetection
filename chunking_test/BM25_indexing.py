import argparse
import sqlite3
import shutil
import time
from pathlib import Path
from typing import Any
from tqdm import tqdm

from pyserini.index.lucene import LuceneIndexer


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def build_bm25_index(
    database_path: str,
    index_path: str,
    batch_size: int = 10_000,
    chunk_type: str = "sentence",
    threads: int = 4,
    overwrite: bool = False,
) -> None:

    database_path = Path(database_path)
    index_path = Path(index_path)

    if chunk_type == "sentence":
        table_name = "sentences"
        id_name="sentence_id"
    elif chunk_type == "word":
        table_name = "chunks"
        id_name="chunk_id"
    else:
        raise ValueError(f"Unsupported chunk type: {chunk_type}")

    if not database_path.exists():
        raise FileNotFoundError(
            f"Input database does not exist: {database_path}"
        )

    if index_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Index directory already exists: {index_path}\n"
                "Use --overwrite only when you intentionally want "
                "to rebuild the index."
            )

        print(f"Removing existing index: {index_path}")
        shutil.rmtree(index_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    indexed_documents = 0


    failed_log_path = (
        index_path.parent / f"{index_path.name}_failed_records.jsonl"
    )

    print("=" * 80)
    print("LUCENE BM25 INDEXING STARTED")
    print("=" * 80)
    print(f"Input file : {  database_path}")
    print(f"Index path : {index_path}")
    print(f"Batch size : {batch_size:,}")
    print(f"Threads    : {threads}")
    print("=" * 80)

    indexer = LuceneIndexer(
        str(index_path),
        threads=threads,
    )

    batch: list[dict[str, str]] = []
    connection = sqlite3.connect(database_path)
    try:
        print("Reading database size...")
        total_rows = connection.execute(
            f"SELECT MAX(rowid) FROM {table_name}"
        ).fetchone()[0]
        print(f"Total rows: {total_rows:,}")

        cursor = connection.execute(
            f"""
            SELECT
                claim_id,
                record_id,
                {id_name},
                contents
            FROM {table_name}
            """
        )

        with tqdm(
            total=total_rows,
            desc="Building BM25 index",
            unit="docs",
        ) as progress_bar:

            while True:
                rows = cursor.fetchmany(batch_size)

                if not rows:
                    break

                for (
                    claim_id,
                    record_id,
                    id_name,
                    contents,
                ) in rows:

                    index_id = (
                        f"{claim_id}_{record_id}_{id_name}"
                    )

                    batch.append(
                        {
                            "id": index_id,
                            "contents": contents,
                        }
                    )

                indexer.add_batch_dict(batch)

                indexed_documents += len(batch)
                progress_bar.update(len(batch))

                batch.clear()

    except Exception:
        raise

    finally:
        indexer.close()
        connection.close()

    

    elapsed = time.time() - start_time


    print()
    print("=" * 80)
    print("LUCENE BM25 INDEXING FINISHED")
    print("=" * 80)
    print(f"Indexed documents    : {indexed_documents:,}")
    print(f"Index path           : {index_path}")
    print(f"Elapsed time         : {format_time(elapsed)}")


    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build a Lucene BM25 index from sentence-level JSONL."
        )
    )

    parser.add_argument(
        "--database_path",
        required=True,
        help="Sentence-level database file.",
    )

    parser.add_argument(
        "--index_path",
        required=True,
        help="Output directory for the Lucene index.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10_000,
        help="Documents submitted to Lucene per batch.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of Lucene indexing threads.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild an existing index directory.",
    )

    parser.add_argument(
        "--chunk_type",
        choices=["sentence", "word"],
        required=True,
        help="As sentence is not the only chunking method, I added this option to specify the chunking method. ",
    )

    args = parser.parse_args()

    build_bm25_index(
        database_path=args.database_path,
        index_path=args.index_path,
        batch_size=args.batch_size,
        threads=args.threads,
        overwrite=args.overwrite,
        chunk_type=args.chunk_type
    )