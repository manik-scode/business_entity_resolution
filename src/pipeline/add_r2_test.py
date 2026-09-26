import sqlite3
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = "results/blocking/blocking_test_v2.db"

conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")
conn.execute("PRAGMA cache_size=-300000")

start = time.time()

print("Adding R2: sorted normalized name...")

max_rowid = conn.execute("SELECT MAX(rowid) FROM s1").fetchone()[0] or 0
batch_size = 100000

for start_id in range(1, max_rowid + 1, batch_size):
    end_id = min(start_id + batch_size - 1, max_rowid)
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
    print(f"Processed S1 rowids {start_id:,} to {end_id:,} ({end_id / max_rowid * 100:.1f}%) | Elapsed: {(time.time() - start) / 60:.2f} min", flush=True)

count = conn.execute(
    "SELECT COUNT(*) FROM candidates"
).fetchone()[0]

print(f"\nTotal candidates after R1 + R2: {count:,}")
print(f"Time: {(time.time() - start) / 60:.2f} minutes")

conn.close()
