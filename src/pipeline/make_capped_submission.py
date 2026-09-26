import csv
import sqlite3
import time

DB = "results/blocking/blocking_test_v2.db"
OUTPUT = "results/matching_results.tsv"

CAP = 50

conn = sqlite3.connect(DB)
conn.execute("PRAGMA cache_size=-300000")

start = time.time()

print(f"Creating submission with max {CAP} candidates per S1...")

with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(
        f,
        delimiter="\t",
        lineterminator="\n"
    )

    writer.writerow([
        "source1_entity_id",
        "matched_entity_ids"
    ])

    # Join s1 with capped candidates (up to CAP per S1),
    # ensuring all 1.73M S1 entities have a row (even if 0 candidates).
    cursor = conn.execute(f"""
        SELECT
            s.entity_id,
            COALESCE(
                (
                    SELECT GROUP_CONCAT(c.candidate_entity_id, ',')
                    FROM (
                        SELECT candidate_entity_id
                        FROM candidates
                        WHERE source1_entity_id = s.entity_id
                        LIMIT {CAP}
                    ) c
                ),
                ''
            )
        FROM s1 s
        ORDER BY s.rowid
    """)

    count = 0

    for s1_id, matched_ids in cursor:
        writer.writerow([
            s1_id,
            matched_ids or ""
        ])
        count += 1
        if count % 200000 == 0:
            print(f"Written: {count:,} | Time: {(time.time() - start) / 60:.1f} min")

print()
print("=" * 70)
print(f"CAPPED SUBMISSION CREATED (CAP = {CAP})")
print("=" * 70)
print(f"S1 rows : {count:,}")
print(f"Output  : {OUTPUT}")
print(f"Done in {(time.time() - start) / 60:.2f} minutes")

conn.close()
