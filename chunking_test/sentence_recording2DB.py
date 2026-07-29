import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm



input_path = Path(r"F:\internal_split\dev_evidence_200.jsonl")
database_path = Path(r"F:\internal_split\dev_sentence.db")


conn = sqlite3.connect(database_path)

conn.execute("""
CREATE TABLE sentences (
    claim_id TEXT,
    record_id INTEGER,
    sentence_id INTEGER,
    source_url TEXT,
    source_type TEXT,
    contents TEXT,
    PRIMARY KEY (claim_id, record_id, sentence_id)
)
""")


record_counter = defaultdict(int)
batch = []


with (input_path.open("r", encoding="utf-8") as file,
      tqdm(
        total=input_path.stat().st_size,
        unit="B",
        unit_scale=True,
        desc="Building database",
    ) as progress_bar,):
    for line in file:
        progress_bar.update(len(line.encode("utf-8")))
        record = json.loads(line)

        claim_id = str(record["claim_id"])
        record_id = record_counter[claim_id]
        source_url = record.get("url", "")
        source_type = record.get("type", "")

        for sentence_id, sentence in enumerate(record["url2text"]):
            sentence = sentence.strip()

            if not sentence:
                continue
            

            batch.append(
                (
                    claim_id,
                    record_id,
                    sentence_id,
                    source_url,
                    source_type,
                    sentence,
                )
            )

            if len(batch) >= 50_000:
                conn.executemany(
                    """
                    INSERT INTO sentences
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    batch,
                )

                conn.commit()
                batch.clear()

        record_counter[claim_id] += 1


if batch:
    conn.executemany(
        """
        INSERT INTO sentences
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        batch,
    )

    conn.commit()


conn.close()

print("Finished")