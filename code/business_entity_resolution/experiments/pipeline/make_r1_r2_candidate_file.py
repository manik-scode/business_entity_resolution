import sqlite3
import csv
import time
from pathlib import Path

DB_PATH = "results/blocking/blocking_test_v2.db"
OUT_PATH = Path("results/candidate_pairs_r1_r2.tsv")

conn = sqlite3.connect(DB_PATH)

start = time.time()

print("Creating complete R1 + R2 candidate_pairs.tsv...")

with open(
    OUT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(
        f,
        delimiter="\t",
        lineterminator="\n"
    )

    writer.writerow([
        "source1_entity_id",
        "candidate_entity_ids"
    ])

    # LEFT JOIN is important:
    # every S1 must have a row, even when it has zero candidates.
    cursor = conn.execute("""
        SELECT
            s.entity_id,
            COALESCE(
                GROUP_CONCAT(c.candidate_entity_id, ','),
                ''
            )
        FROM s1 s
        LEFT JOIN candidates c
            ON c.source1_entity_id = s.entity_id
        GROUP BY s.entity_id
        ORDER BY s.rowid
    """)

    rows = 0
    empty_rows = 0

    for source1_id, candidate_ids in cursor:

        writer.writerow([
            source1_id,
            candidate_ids
        ])

        rows += 1

        if not candidate_ids:
            empty_rows += 1

        if rows % 100_000 == 0:
            print(
                f"Written: {rows:,} | "
                f"Empty: {empty_rows:,} | "
                f"Time: {(time.time() - start)/60:.2f} min"
            )

conn.close()

size = OUT_PATH.stat().st_size

print()
print("=" * 70)
print("R1 + R2 CANDIDATE FILE CREATED")
print("=" * 70)
print(f"S1 rows     : {rows:,}")
print(f"Empty rows  : {empty_rows:,}")
print(f"Non-empty   : {rows - empty_rows:,}")
print(f"Output      : {OUT_PATH}")
print(f"Size        : {size:,} bytes")
print(f"Size MB     : {size / (1024**2):.2f}")
print(f"Time        : {(time.time() - start)/60:.2f} min")
print("=" * 70)