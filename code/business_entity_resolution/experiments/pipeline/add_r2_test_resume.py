import sqlite3
import time

DB_PATH = "results/blocking/blocking_test_v2.db"

conn = sqlite3.connect(DB_PATH, timeout=120)
conn.execute("PRAGMA busy_timeout=120000")
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")

total_s1 = conn.execute(
    "SELECT COUNT(*) FROM s1"
).fetchone()[0]

# Previous run completed approximately up to 500,000
RESUME_FROM = 500001
BATCH_SIZE = 100000

print("Resuming R2: sorted normalized name...")
print(f"S1 rows: {total_s1:,}")
print(f"Starting from rowid: {RESUME_FROM:,}")

overall_start = time.time()

for start_id in range(RESUME_FROM, total_s1 + 1, BATCH_SIZE):

    end_id = min(start_id + BATCH_SIZE - 1, total_s1)

    t0 = time.time()

    conn.execute("""
        INSERT OR IGNORE INTO candidates (
            source1_entity_id,
            candidate_entity_id
        )
        SELECT
            s.entity_id,
            r.entity_id
        FROM s1 s
        JOIN records r
          ON r.country = s.country
         AND r.sorted_name_key = s.sorted_name_key
        WHERE s.sorted_name_key IS NOT NULL
          AND s.sorted_name_key != ''
          AND s.rowid BETWEEN ? AND ?
    """, (start_id, end_id))

    conn.commit()

    elapsed = (time.time() - overall_start) / 60
    progress = end_id / total_s1 * 100

    print(
        f"Processed S1 rowids {start_id:,} to {end_id:,} "
        f"({progress:.1f}%) | "
        f"Batch: {(time.time() - t0)/60:.2f} min | "
        f"Elapsed: {elapsed:.2f} min"
    )

count = conn.execute(
    "SELECT COUNT(*) FROM candidates"
).fetchone()[0]

print()
print("=" * 70)
print("R2 COMPLETE")
print("=" * 70)
print(f"Total candidates R1 + R2: {count:,}")
print(f"Total time: {(time.time() - overall_start)/60:.2f} min")

conn.close()