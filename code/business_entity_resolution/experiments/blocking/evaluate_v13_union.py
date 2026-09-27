import sqlite3
import time

DB = r"results\blocking\blocking_train_v2.db"

db = sqlite3.connect(DB)
db.execute("PRAGMA temp_store=FILE")
db.execute("PRAGMA cache_size=-100000")

start = time.time()

print("Building V13 union count...")

# Existing candidates + R4 + R5
db.execute("DROP TABLE IF EXISTS temp.v13")

db.execute("""
CREATE TEMP TABLE v13(
    s1 TEXT,
    cid TEXT,
    PRIMARY KEY(s1,cid)
)
""")

# Existing
print("Adding existing candidates...")

db.execute("""
INSERT OR IGNORE INTO v13
SELECT source1_entity_id, candidate_entity_id
FROM candidates
""")

print("After existing:",
      db.execute("SELECT COUNT(*) FROM v13").fetchone()[0])

# R4
print("Adding R4...")

db.execute("""
INSERT OR IGNORE INTO v13
SELECT s.entity_id, r.entity_id
FROM s1 s
JOIN records r
  ON r.country = s.country
 AND s.address_number <> ''
 AND s.address_token_sig <> ''
 AND r.address_number = s.address_number
 AND r.address_token_sig = s.address_token_sig
WHERE r.source IN ('S2','S3')
""")

after_r4 = db.execute(
    "SELECT COUNT(*) FROM v13"
).fetchone()[0]

print("After R4:", after_r4)

# R5
print("Adding R5...")

db.execute("""
INSERT OR IGNORE INTO v13
SELECT s.entity_id, r.entity_id
FROM s1 s
JOIN records r
  ON r.country = s.country
 AND s.first_name_token <> ''
 AND s.address_token_sig <> ''
 AND r.first_name_token = s.first_name_token
 AND r.address_token_sig = s.address_token_sig
WHERE r.source IN ('S2','S3')
""")

final_count = db.execute(
    "SELECT COUNT(*) FROM v13"
).fetchone()[0]

print("After R5:", final_count)

# New candidates
existing = db.execute(
    "SELECT COUNT(*) FROM candidates"
).fetchone()[0]

added = final_count - existing

print()
print("=" * 65)
print("V13 UNION")
print("=" * 65)

print(f"Existing candidates : {existing:,}")
print(f"After R4            : {after_r4:,}")
print(f"After R4 + R5       : {final_count:,}")
print(f"NEW candidates      : {added:,}")
print(f"Growth               : {added/existing:.2%}")

# S1 coverage
s1_count = db.execute(
    "SELECT COUNT(DISTINCT s1) FROM v13"
).fetchone()[0]

print(f"S1 with candidates   : {s1_count:,}")

print(f"Runtime              : {time.time()-start:.1f}s")
print("=" * 65)

db.close()