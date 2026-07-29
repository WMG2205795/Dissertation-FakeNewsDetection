import sqlite3
from pathlib import Path

DB_PATH = Path(r"F:\internal_split\dev_sentence.db")
TABLE_NAME = "sentences"

connection = sqlite3.connect(DB_PATH)

total = connection.execute(
    f"""
    SELECT COUNT(*)
    FROM {TABLE_NAME}
    """
).fetchone()[0]

unique = connection.execute(
    f"""
    SELECT COUNT(DISTINCT contents)
    FROM {TABLE_NAME}
    WHERE contents IS NOT NULL
    """
).fetchone()[0]

duplicate_count = total - unique

print("Total rows:", total)
print("Unique contents:", unique)
print("Duplicate rows:", duplicate_count)
print(
    "Duplicate ratio:",
    duplicate_count / total
    if total else 0
)

connection.close()