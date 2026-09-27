import csv
import sqlite3
import time

DB = "results/blocking/blocking_test_v2.db"
OUTPUT = "results/matching_results_r1.tsv"

conn = sqlite3.connect(DB)
conn.execute("PRAGMA cache_size=-300000")
conn.execute("PRAGMA temp_store=MEMORY")

start = time.time()

print("Creating R1-only matching_results.tsv...")

with open(
    OUTPUT,
    "w",
    encoding="utf-8",
    newline=""
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

    # candidates table has PRIMARY KEY
    # (source1_entity_id, candidate_entity_id),
    # so rows are already unique.
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

    count = 0

    for s1_id, matched_ids in cursor:

        writer.writerow([
            s1_id,
            matched_ids or ""
        ])

        count += 1

        if count % 100000 == 0:
            print(
                f"Written: {count:,} | "
                f"Time: {(time.time() - start) / 60:.1f} min"
            )

conn.close()

print()
print("=" * 70)
print("R1 SUBMISSION CREATED")
print("=" * 70)
print(f"S1 rows : {count:,}")
print(f"Output  : {OUTPUT}")
print(f"Time    : {(time.time() - start) / 60:.2f} min")
