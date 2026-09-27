import sqlite3
import time

DB = r"results\blocking\blocking_train_v2.db"

db = sqlite3.connect(DB)
db.execute("PRAGMA temp_store=FILE")
db.execute("PRAGMA cache_size=-100000")

start = time.time()

# Current candidates
base = db.execute(
    "SELECT COUNT(*) FROM candidates"
).fetchone()[0]

print(f"Existing candidates: {base:,}")

# ---------------------------------------------------------
# Create indexes for composite blocking keys
# ---------------------------------------------------------

print("Creating temporary indexes...")

db.execute("""
CREATE INDEX IF NOT EXISTS idx_records_country_num_token
ON records(country, address_number, address_token_sig)
""")

db.execute("""
CREATE INDEX IF NOT EXISTS idx_records_country_first_token
ON records(country, first_name_token, address_token_sig)
""")

db.commit()

# ---------------------------------------------------------
# R4: address number + address token
# ---------------------------------------------------------

print("Testing R4: country + address_number + address_token_sig")

r4 = db.execute("""
SELECT COUNT(*)
FROM s1 s
JOIN records r
  ON r.country = s.country
 AND s.address_number <> ''
 AND s.address_token_sig <> ''
 AND r.address_number = s.address_number
 AND r.address_token_sig = s.address_token_sig
WHERE r.source IN ('S2','S3')
""").fetchone()[0]

print(f"R4 raw pairs: {r4:,}")

# ---------------------------------------------------------
# R5: first name + address token
# ---------------------------------------------------------

print("Testing R5: country + first_name_token + address_token_sig")

r5 = db.execute("""
SELECT COUNT(*)
FROM s1 s
JOIN records r
  ON r.country = s.country
 AND s.first_name_token <> ''
 AND s.address_token_sig <> ''
 AND r.first_name_token = s.first_name_token
 AND r.address_token_sig = s.address_token_sig
WHERE r.source IN ('S2','S3')
""").fetchone()[0]

print(f"R5 raw pairs: {r5:,}")

# ---------------------------------------------------------
# Evaluate how many currently MISSED true pairs each rule
# would recover.
# ---------------------------------------------------------

print("Loading ground truth...")

db.execute("DROP TABLE IF EXISTS temp.truth")

db.execute("""
CREATE TEMP TABLE truth(
    s1 TEXT,
    cid TEXT,
    PRIMARY KEY(s1,cid)
)
""")

import csv

with open(
    r"student_resource\dataset\train\train_ground_truth.tsv",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f, delimiter="\t")

    # Only S1s in this experiment
    valid_s1 = {
        x[0]
        for x in db.execute("SELECT DISTINCT source1_entity_id FROM candidates")
    }

    batch = []

    for row in reader:

        s1 = row["source1_entity_id"]

        if s1 not in valid_s1:
            continue

        ids = row["matched_entity_ids"].strip()

        if not ids:
            continue

        for cid in ids.split(","):
            cid = cid.strip()

            if cid:
                batch.append((s1, cid))

        if len(batch) >= 100000:
            db.executemany(
                "INSERT OR IGNORE INTO truth VALUES (?,?)",
                batch
            )
            batch.clear()

    if batch:
        db.executemany(
            "INSERT OR IGNORE INTO truth VALUES (?,?)",
            batch
        )

db.execute("CREATE INDEX idx_truth ON truth(s1,cid)")

# Current candidates
db.execute("""
CREATE TEMP TABLE missed AS
SELECT t.s1, t.cid
FROM truth t
LEFT JOIN candidates c
  ON c.source1_entity_id = t.s1
 AND c.candidate_entity_id = t.cid
WHERE c.candidate_entity_id IS NULL
""")

db.execute("CREATE INDEX idx_missed ON missed(s1,cid)")

missed_count = db.execute(
    "SELECT COUNT(*) FROM missed"
).fetchone()[0]

print(f"Currently missed true pairs: {missed_count:,}")

# ---------------------------------------------------------
# R4 recovery
# ---------------------------------------------------------

r4_recovered = db.execute("""
SELECT COUNT(*)
FROM missed m
JOIN s1 s
  ON s.entity_id = m.s1
JOIN records r
  ON r.entity_id = m.cid
WHERE r.country = s.country
  AND s.address_number <> ''
  AND s.address_token_sig <> ''
  AND r.address_number = s.address_number
  AND r.address_token_sig = s.address_token_sig
""").fetchone()[0]

# ---------------------------------------------------------
# R5 recovery
# ---------------------------------------------------------

r5_recovered = db.execute("""
SELECT COUNT(*)
FROM missed m
JOIN s1 s
  ON s.entity_id = m.s1
JOIN records r
  ON r.entity_id = m.cid
WHERE r.country = s.country
  AND s.first_name_token <> ''
  AND s.address_token_sig <> ''
  AND r.first_name_token = s.first_name_token
  AND r.address_token_sig = s.address_token_sig
""").fetchone()[0]

# ---------------------------------------------------------
# Combined recovery
# ---------------------------------------------------------

combined = db.execute("""
SELECT COUNT(*)
FROM missed m
JOIN s1 s
  ON s.entity_id = m.s1
JOIN records r
  ON r.entity_id = m.cid
WHERE r.country = s.country
AND (
    (
        s.address_number <> ''
        AND s.address_token_sig <> ''
        AND r.address_number = s.address_number
        AND r.address_token_sig = s.address_token_sig
    )
    OR
    (
        s.first_name_token <> ''
        AND s.address_token_sig <> ''
        AND r.first_name_token = s.first_name_token
        AND r.address_token_sig = s.address_token_sig
    )
)
""").fetchone()[0]

print()
print("=" * 65)
print("V13 BLOCKING EXPERIMENT")
print("=" * 65)

print(f"Existing candidates       : {base:,}")
print(f"Currently missed pairs    : {missed_count:,}")
print()
print(f"R4 raw candidate pairs    : {r4:,}")
print(f"R4 missed recovered       : {r4_recovered:,}")
print(f"R4 recovery of missed    : {r4_recovered/missed_count:.2%}")
print()
print(f"R5 raw candidate pairs    : {r5:,}")
print(f"R5 missed recovered       : {r5_recovered:,}")
print(f"R5 recovery of missed    : {r5_recovered/missed_count:.2%}")
print()
print(f"Combined missed recovered : {combined:,}")
print(f"Combined recovery         : {combined/missed_count:.2%}")
print()
print(f"Runtime                   : {time.time()-start:.1f}s")
print("=" * 65)

db.close()