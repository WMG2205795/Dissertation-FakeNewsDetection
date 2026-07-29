import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


input_path = Path(r"F:\internal_split\dev_evidence_200.jsonl")
output_dir = Path(r"F:\internal_split")

CHUNK_CONFIGS = [
    {
        "name": "word_100_overlap_25",
        "chunk_words": 100,
        "overlap_words": 25,
    },
    {
        "name": "word_200_overlap_50",
        "chunk_words": 200,
        "overlap_words": 50,
    },
    {
        "name": "word_400_overlap_100",
        "chunk_words": 400,
        "overlap_words": 100,
    },
]

BATCH_SIZE = 50_000


def word_chunking_with_overlap(
        text: str,
        chunk_words: int,
        overlap_words: int,
    ):

        words = text.split()

        if not words:
            return

        stride = chunk_words - overlap_words
        chunk_id = 0
        word_start = 0

        while word_start < len(words):
            word_end = min(
                word_start + chunk_words,
                len(words),
            )

            chunk_words_list = words[word_start:word_end]
            chunk_text = " ".join(chunk_words_list)

            if chunk_text.strip():
                yield (
                    chunk_id,
                    word_start,
                    word_end,
                    chunk_text,
                )

            if word_end >= len(words):
                break

            word_start += stride
            chunk_id += 1


for config in CHUNK_CONFIGS:
    CHUNK_WORDS = config["chunk_words"]
    OVERLAP_WORDS = config["overlap_words"]

    database_name = f"dev_chunks_{CHUNK_WORDS}_overlap_{OVERLAP_WORDS}.db"
    database_path = output_dir / database_name

    if CHUNK_WORDS <= 0:
        raise ValueError("CHUNK_WORDS must be greater than 0.")

    if OVERLAP_WORDS < 0:
        raise ValueError("OVERLAP_WORDS cannot be negative.")

    if OVERLAP_WORDS >= CHUNK_WORDS:
        raise ValueError(
            "OVERLAP_WORDS must be smaller than CHUNK_WORDS."
        )

    STRIDE_WORDS = CHUNK_WORDS - OVERLAP_WORDS

    conn = sqlite3.connect(database_path)

    conn.execute(
        """
        CREATE TABLE chunks (
            claim_id TEXT,
            record_id INTEGER,
            chunk_id INTEGER,

            source_url TEXT,
            source_type TEXT,

            word_start INTEGER,
            word_end INTEGER,
            word_count INTEGER,

            chunk_words INTEGER,
            overlap_words INTEGER,
            stride_words INTEGER,

            contents TEXT,

            PRIMARY KEY (
                claim_id,
                record_id,
                chunk_id
            )
        )
        """
    )


    record_counter = defaultdict(int)
    batch = []

    document_count = 0
    chunk_count = 0
    empty_document_count = 0


    with (
        input_path.open("r", encoding="utf-8") as file,
        tqdm(
            total=input_path.stat().st_size,
            unit="B",
            unit_scale=True,
            desc="Building word-chunk database",
        ) as progress_bar,
    ):
        for line in file:
            progress_bar.update(len(line.encode("utf-8")))

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            claim_id = str(record["claim_id"])

            record_id = record_counter[claim_id]

            source_url = record.get("url", "")
            source_type = record.get("type", "")

            """
            originally, the article is recorded as a list of sentences,
            but now I treat the whole article as a single document for word chunking.
            """
            sentences = [
                sentence.strip()
                for sentence in record.get("url2text", [])
                if isinstance(sentence, str)
                and sentence.strip()
            ]

            document_text = " ".join(sentences)

            document_count += 1

            if not document_text:
                empty_document_count += 1
                record_counter[claim_id] += 1
                continue
            for (
                chunk_id,
                word_start,
                word_end,
                chunk_text,
            ) in word_chunking_with_overlap(
                text=document_text,
                chunk_words=CHUNK_WORDS,
                overlap_words=OVERLAP_WORDS,
            ):
                word_count = word_end - word_start

                batch.append(
                    (
                        claim_id,
                        record_id,
                        chunk_id,

                        source_url,
                        source_type,

                        word_start,
                        word_end,
                        word_count,

                        CHUNK_WORDS,
                        OVERLAP_WORDS,
                        STRIDE_WORDS,

                        chunk_text,
                    )
                )

                chunk_count += 1

                if len(batch) >= BATCH_SIZE:
                    conn.executemany(
                        """
                        INSERT INTO chunks (
                            claim_id,
                            record_id,
                            chunk_id,

                            source_url,
                            source_type,

                            word_start,
                            word_end,
                            word_count,

                            chunk_words,
                            overlap_words,
                            stride_words,

                            contents
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch,
                    )

                    conn.commit()
                    batch.clear()
            record_counter[claim_id] += 1

    if batch:
        conn.executemany(
            """
            INSERT INTO chunks (
                claim_id,
                record_id,
                chunk_id,

                source_url,
                source_type,

                word_start,
                word_end,
                word_count,

                chunk_words,
                overlap_words,
                stride_words,

                contents
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )

        conn.commit()

    print("Creating SQLite indexes...")

    conn.execute(
        """
        CREATE INDEX idx_chunks_claim_id
        ON chunks(claim_id)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_chunks_source_url
        ON chunks(source_url)
        """
    )

    conn.commit()


    conn.execute(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )

    metadata = [
        ("chunking_method", "fixed_word_with_overlap"),
        ("chunk_words", str(CHUNK_WORDS)),
        ("overlap_words", str(OVERLAP_WORDS)),
        ("stride_words", str(STRIDE_WORDS)),
        ("input_path", str(input_path)),
    ]

    conn.executemany(
        """
        INSERT INTO metadata (key, value)
        VALUES (?, ?)
        """,
        metadata,
    )

    conn.commit()
    conn.close()


    print("Finished")
    print(f"Documents processed: {document_count:,}")
    print(f"Chunks generated: {chunk_count:,}")
    print(f"Empty documents skipped: {empty_document_count:,}")
    print(f"Chunk words: {CHUNK_WORDS}")
    print(f"Overlap words: {OVERLAP_WORDS}")
    print(f"Stride words: {STRIDE_WORDS}")
    print(f"Database saved to: {database_path}")