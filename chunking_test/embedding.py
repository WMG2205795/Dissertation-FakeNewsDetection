import argparse
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm



MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 256
# This is for checkpoint
SHARD_SIZE = 100_000
SAVE_DTYPE = np.float32


DATABASE_SCHEMAS = {
    "sentence": {
        "table": "sentences",
        "text_column": "contents",
        "id_column": "sentence_id",
    },
    "word": {
        "table": "chunks",
        "text_column": "contents",
        "id_column": "chunks_id",
    },
}


def atomic_save_numpy(file_path: Path, array: np.ndarray) -> None:
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")

    with open(temp_path, "wb") as file:
        np.save(file, array)

    os.replace(temp_path, file_path)


def atomic_save_json(file_path: Path, data: dict) -> None:
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    os.replace(temp_path, file_path)


def atomic_save_keys(file_path: Path, keys: list[tuple]) -> None:
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        for key in keys:
            record = {
                "claim_id": key[0],
                "record_id": key[1],
                "chunk_id": key[2],
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    os.replace(temp_path, file_path)


def load_checkpoint(checkpoint_path: Path) -> dict:
    if not checkpoint_path.exists():
        return {
            "last_rowid": 0,
            "next_shard_id": 0,
            "processed_count": 0,
        }

    with open(checkpoint_path, "r", encoding="utf-8") as file:
        return json.load(file)


def fetch_batch(
    connection: sqlite3.Connection,
    mode: str,
    last_rowid: int,
    limit: int,
) -> list[tuple]:


    schema = DATABASE_SCHEMAS[mode]

    table = schema["table"]
    text_column = schema["text_column"]
    id_column = schema["id_column"]

    query = f"""
        SELECT
            rowid,
            claim_id,
            record_id,
            {id_column},
            {text_column}
        FROM {table}
        WHERE rowid > ?
        ORDER BY rowid
        LIMIT ?
    """

    cursor = connection.execute(query, (last_rowid, limit))
    return cursor.fetchall()



def load_embedding_model() -> SentenceTransformer:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    if device == "cuda":
        model.half()

    return model



def build_embeddings(
    db_path: Path,
    output_dir: Path,
    mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_path)

    last_rowid = checkpoint["last_rowid"]
    shard_id = checkpoint["next_shard_id"]
    processed_count = checkpoint["processed_count"]

    print(f"Mode: {mode}")
    print(f"Database: {db_path}")
    print(f"Output directory: {output_dir}")
    print(f"Resume after rowid: {last_rowid}")
    print(f"Already processed: {processed_count:,}")

    connection = sqlite3.connect(str(db_path))

    connection.execute("PRAGMA query_only = ON")


    model = load_embedding_model()

    try:
        while True:
            rows = fetch_batch(
                connection=connection,
                mode=mode,
                last_rowid=last_rowid,
                limit=SHARD_SIZE,
            )

            if not rows:
                break

            texts = []
            keys = []

            for row in rows:
                _, claim_id, record_id, chunk_id, text = row

                if text is None:
                    text = ""

                texts.append(str(text))
                keys.append((claim_id, record_id, chunk_id))

            embeddings = model.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )

            embeddings = embeddings.astype(
                SAVE_DTYPE,
                copy=False,
            )

            vector_path = output_dir / f"vectors_{shard_id:05d}.npy"
            keys_path = output_dir / f"keys_{shard_id:05d}.jsonl"
            atomic_save_numpy(vector_path, embeddings)
            atomic_save_keys(keys_path, keys)
            last_rowid = rows[-1][0]
            processed_count += len(rows)
            shard_id += 1

            checkpoint = {
                "database": str(db_path.resolve()),
                "mode": mode,
                "model_name": MODEL_NAME,
                "last_rowid": last_rowid,
                "next_shard_id": shard_id,
                "processed_count": processed_count,
                "embedding_dimension": int(embeddings.shape[1]),
                "save_dtype": str(SAVE_DTYPE),
            }

            atomic_save_json(checkpoint_path, checkpoint)

            print(
                f"\nSaved shard {shard_id - 1}: "
                f"{len(rows):,} vectors, "
                f"last rowid = {last_rowid}"
            )

    finally:
        connection.close()

    print("\nEmbedding completed.")
    print(f"Total processed: {processed_count:,}")
    print(f"Output directory: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BGE embeddings from a SQLite chunk database."
    )

    parser.add_argument(
        "--database_path",
        type=Path,
        required=True,
        help="Path to the SQLite database.",
    )

    parser.add_argument(
        "--embedding_path",
        type=Path,
        required=True,
        help="Directory used to save vectors, keys and checkpoint.",
    )

    parser.add_argument(
        "--chunk_type",
        choices=["sentence", "word"],
        required=True,
        help=(
            "sentence: key=(claim_id, record_id, sentence_id); "
            "word: key=(claim_id, record_id, chunks_id)"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    build_embeddings(
        db_path=args.database_path,
        output_dir=args.embedding_path,
        mode=args.chunk_type,
    )