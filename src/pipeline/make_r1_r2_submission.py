import sqlite3
import csv
import time
from pathlib import Path

DB_PATH = "results/blocking/blocking_test_v2.db"
OUT_DIR = Path("results")

MATCHING_OUT = OUT_DIR / "matching_results_r1_r2.tsv"
CANDIDATE_OUT = OUT_DIR / "candidate_pairs_r1_r2.tsv"

conn = sqlite3.connect(DB_PATH)

start = time.time()

print("Creating R1 + R2 submission...")

# ---------------------------------------------------------
# Candidate pairs
# ---------------------------------------------------------

with open(CANDIDATE_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")

    writer.writerow([
        "source1_entity_id",
        "candidate_entity_id"
    ])

    cur = conn.execute("""
        SELECT
            source1_entity_id,
            candidate_entity_id
        FROM candidates
        ORDER BY source1_entity_id, candidate_entity_id
    """)

    count = 0

    for row in cur:
        writer.writerow(row)
        count += 1

        if count % 1_000_000 == 0:
            print(
                f"Candidate pairs written: {count:,} | "
                f"Time: {(time.time() - start)/60:.2f} min"
            )

print(f"Candidate pairs: {count:,}")


# ---------------------------------------------------------
# Matching results
# ---------------------------------------------------------

print("Creating matching_results...")

with open(MATCHING_OUT, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f, delimiter="\t")

    writer.writerow([
        "source1_entity_id",
        "matched_entity_ids"
    ])

    cur = conn.execute("""
        SELECT
            s.entity_id,
            GROUP_CONCAT(c.candidate_entity_id, ',')
        FROM s1 s
        LEFT JOIN candidates c
            ON c.source1_entity_id = s.entity_id
        GROUP BY s.entity_id
        ORDER BY s.rowid
    """)

    rows = 0

    for entity_id, matches in cur:

        writer.writerow([
            entity_id,
            matches if matches else ""
        ])

        rows += 1

        if rows % 100_000 == 0:
            print(
                f"Matching rows written: {rows:,} | "
                f"Time: {(time.time() - start)/60:.2f} min"
            )

print()
print("=" * 70)
print("R1 + R2 SUBMISSION CREATED")
print("=" * 70)
print(f"S1 rows       : {rows:,}")
print(f"Candidates    : {count:,}")
print(f"Matching file : {MATCHING_OUT}")
print(f"Candidate file: {CANDIDATE_OUT}")
print(f"Time          : {(time.time() - start)/60:.2f} min")

conn.close()