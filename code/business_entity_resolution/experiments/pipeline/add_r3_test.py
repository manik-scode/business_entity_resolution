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

BATCH_SIZE = 100_000

print("=" * 70)
print("Adding R3: name_token + address_number")
print("=" * 70)
print(f"S1 rows: {total_s1:,}")
print()

start_time = time.time()

for start_id in range(1, total_s1 + 1, BATCH_SIZE):

    end_id = min(
        start_id + BATCH_SIZE - 1,
        total_s1
    )

    batch_start = time.time()

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
         AND r.name_token_key = s.name_token_key
         AND r.address_number = s.address_number
        WHERE s.name_token_key IS NOT NULL
          AND s.name_token_key != ''
          AND s.address_number IS NOT NULL
          AND s.address_number != ''
          AND r.name_token_key IS NOT NULL
          AND r.name_token_key != ''
          AND r.address_number IS NOT NULL
          AND r.address_number != ''
          AND s.rowid BETWEEN ? AND ?
    """, (start_id, end_id))

    conn.commit()

    elapsed = (time.time() - start_time) / 60
    batch_time = (time.time() - batch_start) / 60
    progress = end_id / total_s1 * 100

    print(
        f"Processed S1 rowids {start_id:,} to {end_id:,} "
        f"({progress:.1f}%) | "
        f"Batch: {batch_time:.2f} min | "
        f"Elapsed: {elapsed:.2f} min"
    )

total_candidates = conn.execute(
    "SELECT COUNT(*) FROM candidates"
).fetchone()[0]

conn.close()

print()
print("=" * 70)
print("R3 COMPLETE")
print("=" * 70)
print(f"Total candidates R1 + R2 + R3: {total_candidates:,}")
print(f"Total time: {(time.time() - start_time)/60:.2f} min")
print("=" * 70)