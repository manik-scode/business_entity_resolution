import sqlite3
import csv
import os
import time

DB_PATH = "results/blocking/blocking_test_v2.db"
OUT_MATCHING = "results/matching_results_r1_r2_r3_cap50.tsv"

CAP = 50

print("=" * 70)
print("Generating R1 + R2 + R3 matching submission")
print("=" * 70)

start = time.time()

conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA busy_timeout=120000")

cur = conn.cursor()

total_s1 = cur.execute(
    "SELECT COUNT(*) FROM s1"
).fetchone()[0]

print(f"S1 rows: {total_s1:,}")
print(f"Candidate cap per S1: {CAP}")
print()

os.makedirs(os.path.dirname(OUT_MATCHING), exist_ok=True)

with open(OUT_MATCHING, "w", encoding="utf-8", newline="") as f:

    writer = csv.writer(
        f,
        delimiter="\t",
        lineterminator="\n"
    )

    writer.writerow([
        "source1_entity_id",
        "matched_entity_ids"
    ])

    processed = 0

    for row in cur.execute("""
        SELECT
            s.entity_id,
            c.candidate_entity_id
        FROM s1 s
        LEFT JOIN candidates c
            ON s.entity_id = c.source1_entity_id
        ORDER BY s.rowid, c.candidate_entity_id
    """):

        s1_id = row[0]

        if processed == 0:
            current_id = s1_id
            matches = []

        if s1_id != current_id:

            writer.writerow([
                current_id,
                ",".join(matches[:CAP])
            ])

            current_id = s1_id
            matches = []

        if row[1] is not None:
            matches.append(row[1])

        processed += 1

        if processed % 5_000_000 == 0:
            elapsed = (time.time() - start) / 60
            print(
                f"Processed joined rows: {processed:,} "
                f"| Elapsed: {elapsed:.2f} min"
            )

    # write final S1
    if processed > 0:
        writer.writerow([
            current_id,
            ",".join(matches[:CAP])
        ])

conn.close()

elapsed = (time.time() - start) / 60
size_mb = os.path.getsize(OUT_MATCHING) / (1024 * 1024)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
print(f"Output: {OUT_MATCHING}")
print(f"File size: {size_mb:.2f} MB")
print(f"Time: {elapsed:.2f} min")
print("=" * 70)