import sqlite3
import csv
import time
from pathlib import Path

DB_PATH = "results/blocking/blocking_test_v2.db"
OUT_PATH = Path("results/matching_results_r1_r2_cap50.tsv")

CAP = 50

conn = sqlite3.connect(DB_PATH)

start = time.time()

print("=" * 70)
print("Creating R1 + R2 capped submission")
print("=" * 70)
print(f"Per-S1 match cap: {CAP}")
print()

# ------------------------------------------------------------------
# Important:
# We DO NOT modify the existing candidates table.
# We only read from it and create a new submission.
# ------------------------------------------------------------------

query = """
SELECT
    s.entity_id,
    c.candidate_entity_id
FROM s1 s
LEFT JOIN (
    SELECT
        source1_entity_id,
        candidate_entity_id,
        ROW_NUMBER() OVER (
            PARTITION BY source1_entity_id
            ORDER BY candidate_entity_id
        ) AS rn
    FROM candidates
) c
    ON c.source1_entity_id = s.entity_id
   AND c.rn <= ?
ORDER BY s.rowid
"""

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

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
        "matched_entity_ids"
    ])

    cursor = conn.execute(query, (CAP,))

    current_s1 = None
    matches = []
    rows_written = 0

    for entity_id, candidate_id in cursor:

        # New S1 entity
        if current_s1 is not None and entity_id != current_s1:

            writer.writerow([
                current_s1,
                ",".join(matches)
            ])

            rows_written += 1

            if rows_written % 100_000 == 0:
                print(
                    f"Written: {rows_written:,} | "
                    f"Time: {(time.time() - start)/60:.2f} min"
                )

            matches = []

        current_s1 = entity_id

        if candidate_id is not None:
            matches.append(candidate_id)

    # Write final S1
    if current_s1 is not None:
        writer.writerow([
            current_s1,
            ",".join(matches)
        ])
        rows_written += 1


conn.close()

file_size = OUT_PATH.stat().st_size

print()
print("=" * 70)
print("R1 + R2 CAP-50 SUBMISSION CREATED")
print("=" * 70)
print(f"S1 rows     : {rows_written:,}")
print(f"Match cap   : {CAP}")
print(f"Output      : {OUT_PATH}")
print(f"File size   : {file_size:,} bytes")
print(f"File size   : {file_size / (1024**2):.2f} MB")
print(f"Time        : {(time.time() - start)/60:.2f} min")
print("=" * 70)