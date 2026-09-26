import sqlite3
import time

DB = r"results\blocking\blocking_train_v2.db"

db = sqlite3.connect(DB)
db.execute("PRAGMA temp_store=FILE")
db.execute("PRAGMA cache_size=-100000")

start = time.time()

# ---------------------------------------------------------
# Build frequency tables
# ---------------------------------------------------------

print("Building R4 frequency table...")

db.execute("DROP TABLE IF EXISTS temp.r4_freq")

db.execute("""
CREATE TEMP TABLE r4_freq AS
SELECT
    country,
    address_number,
    address_token_sig,
    COUNT(*) AS freq
FROM records
WHERE source IN ('S2','S3')
  AND country <> ''
  AND address_number <> ''
  AND address_token_sig <> ''
GROUP BY country, address_number, address_token_sig
""")

db.execute("""
CREATE INDEX idx_r4_freq
ON r4_freq(country,address_number,address_token_sig)
""")

print("R4 keys:",
      db.execute("SELECT COUNT(*) FROM r4_freq").fetchone()[0])

print("Building R5 frequency table...")

db.execute("DROP TABLE IF EXISTS temp.r5_freq")

db.execute("""
CREATE TEMP TABLE r5_freq AS
SELECT
    country,
    first_name_token,
    address_token_sig,
    COUNT(*) AS freq
FROM records
WHERE source IN ('S2','S3')
  AND country <> ''
  AND first_name_token <> ''
  AND address_token_sig <> ''
GROUP BY country, first_name_token, address_token_sig
""")

db.execute("""
CREATE INDEX idx_r5_freq
ON r5_freq(country,first_name_token,address_token_sig)
""")

print("R5 keys:",
      db.execute("SELECT COUNT(*) FROM r5_freq").fetchone()[0])

# ---------------------------------------------------------
# Ground truth
# ---------------------------------------------------------

print("Loading ground truth...")

import csv

valid_s1 = {
    x[0]
    for x in db.execute(
        "SELECT DISTINCT source1_entity_id FROM candidates"
    )
}

db.execute("DROP TABLE IF EXISTS temp.truth")

db.execute("""
CREATE TEMP TABLE truth(
    s1 TEXT,
    cid TEXT,
    PRIMARY KEY(s1,cid)
)
""")

batch = []

with open(
    r"student_resource\dataset\train\train_ground_truth.tsv",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f, delimiter="\t")

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
                batch.append((s1,cid))

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

# ---------------------------------------------------------
# Current candidates
# ---------------------------------------------------------

db.execute("DROP TABLE IF EXISTS temp.missed")

db.execute("""
CREATE TEMP TABLE missed AS
SELECT t.s1,t.cid
FROM truth t
LEFT JOIN candidates c
  ON c.source1_entity_id=t.s1
 AND c.candidate_entity_id=t.cid
WHERE c.candidate_entity_id IS NULL
""")

db.execute("CREATE INDEX idx_missed ON missed(s1,cid)")

missed = db.execute(
    "SELECT COUNT(*) FROM missed"
).fetchone()[0]

print(f"Missed true pairs: {missed:,}")

# ---------------------------------------------------------
# Test caps
# ---------------------------------------------------------

caps = [25,50,100,250,500,1000]

print()
print("=" * 95)
print(
    f"{'CAP':>8} "
    f"{'R4 NEW':>14} "
    f"{'R4 REC':>12} "
    f"{'R5 NEW':>14} "
    f"{'R5 REC':>12} "
    f"{'COMBINED REC':>15}"
)
print("=" * 95)

for cap in caps:

    # R4 candidate count
    r4_new = db.execute("""
        SELECT COUNT(*)
        FROM s1 s
        JOIN r4_freq f
          ON f.country=s.country
         AND f.address_number=s.address_number
         AND f.address_token_sig=s.address_token_sig
        JOIN records r
          ON r.country=s.country
         AND r.address_number=s.address_number
         AND r.address_token_sig=s.address_token_sig
         AND r.source IN ('S2','S3')
        LEFT JOIN candidates c
          ON c.source1_entity_id=s.entity_id
         AND c.candidate_entity_id=r.entity_id
        WHERE f.freq <= ?
          AND c.candidate_entity_id IS NULL
    """,(cap,)).fetchone()[0]

    # R4 recovered true
    r4_rec = db.execute("""
        SELECT COUNT(*)
        FROM missed m
        JOIN s1 s ON s.entity_id=m.s1
        JOIN records r ON r.entity_id=m.cid
        JOIN r4_freq f
          ON f.country=s.country
         AND f.address_number=s.address_number
         AND f.address_token_sig=s.address_token_sig
        WHERE f.freq <= ?
    """,(cap,)).fetchone()[0]

    # R5 candidate count
    r5_new = db.execute("""
        SELECT COUNT(*)
        FROM s1 s
        JOIN r5_freq f
          ON f.country=s.country
         AND f.first_name_token=s.first_name_token
         AND f.address_token_sig=s.address_token_sig
        JOIN records r
          ON r.country=s.country
         AND r.first_name_token=s.first_name_token
         AND r.address_token_sig=s.address_token_sig
         AND r.source IN ('S2','S3')
        LEFT JOIN candidates c
          ON c.source1_entity_id=s.entity_id
         AND c.candidate_entity_id=r.entity_id
        WHERE f.freq <= ?
          AND c.candidate_entity_id IS NULL
    """,(cap,)).fetchone()[0]

    # R5 recovered
    r5_rec = db.execute("""
        SELECT COUNT(*)
        FROM missed m
        JOIN s1 s ON s.entity_id=m.s1
        JOIN records r ON r.entity_id=m.cid
        JOIN r5_freq f
          ON f.country=s.country
         AND f.first_name_token=s.first_name_token
         AND f.address_token_sig=s.address_token_sig
        WHERE f.freq <= ?
    """,(cap,)).fetchone()[0]

    # Combined recovery
    combined = db.execute("""
        SELECT COUNT(*)
        FROM missed m
        JOIN s1 s ON s.entity_id=m.s1
        JOIN records r ON r.entity_id=m.cid
        LEFT JOIN r4_freq f4
          ON f4.country=s.country
         AND f4.address_number=s.address_number
         AND f4.address_token_sig=s.address_token_sig
         AND f4.freq <= ?
        LEFT JOIN r5_freq f5
          ON f5.country=s.country
         AND f5.first_name_token=s.first_name_token
         AND f5.address_token_sig=s.address_token_sig
         AND f5.freq <= ?
        WHERE f4.freq IS NOT NULL
           OR f5.freq IS NOT NULL
    """,(cap,cap)).fetchone()[0]

    print(
        f"{cap:>8} "
        f"{r4_new:>14,} "
        f"{r4_rec:>12,} "
        f"{r5_new:>14,} "
        f"{r5_rec:>12,} "
        f"{combined:>15,}"
    )

print("=" * 95)
print(f"Runtime: {time.time()-start:.1f}s")

db.close()