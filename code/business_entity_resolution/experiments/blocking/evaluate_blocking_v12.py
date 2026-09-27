import csv
import sqlite3
import time
from collections import Counter

DB_PATH = r"results\blocking\blocking_train_v2.db"
GT_PATH = r"student_resource\dataset\train\train_ground_truth.tsv"

start = time.time()

db = sqlite3.connect(DB_PATH)
db.execute("PRAGMA temp_store=FILE")
db.execute("PRAGMA cache_size=-100000")

print("Loading candidates...")

# Candidate table -> temporary indexed table
db.execute("DROP TABLE IF EXISTS temp.cand")
db.execute("""
    CREATE TEMP TABLE cand(
        s1 TEXT,
        cid TEXT,
        PRIMARY KEY(s1, cid)
    )
""")

cur = db.execute("""
    SELECT source1_entity_id, candidate_entity_id
    FROM candidates
""")

batch = []

for row in cur:
    batch.append(row)

    if len(batch) >= 100000:
        db.executemany(
            "INSERT OR IGNORE INTO cand VALUES (?, ?)",
            batch
        )
        batch.clear()

if batch:
    db.executemany(
        "INSERT OR IGNORE INTO cand VALUES (?, ?)",
        batch
    )

db.execute("CREATE INDEX idx_temp_cand_s1 ON cand(s1)")

candidate_count = db.execute(
    "SELECT COUNT(*) FROM cand"
).fetchone()[0]

candidate_s1 = {
    row[0]
    for row in db.execute(
        "SELECT DISTINCT s1 FROM cand"
    )
}

print(f"Candidates: {candidate_count:,}")
print(f"Candidate S1: {len(candidate_s1):,}")

# ---------------------------------------------------------
# Load TRUE pairs only for these 100k S1 records
# ---------------------------------------------------------

print("Reading ground truth...")

true_pairs = []
true_by_s1 = Counter()

with open(
    GT_PATH,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(f, delimiter="\t")

    for row in reader:

        s1 = row["source1_entity_id"]

        if s1 not in candidate_s1:
            continue

        matched = row["matched_entity_ids"].strip()

        if not matched:
            continue

        for cid in matched.split(","):

            cid = cid.strip()

            if cid:
                true_pairs.append((s1, cid))
                true_by_s1[s1] += 1


print(f"True pairs in this 100k subset: {len(true_pairs):,}")
print(f"S1 having at least one true match: {len(true_by_s1):,}")

# ---------------------------------------------------------
# Put truth into SQLite
# ---------------------------------------------------------

db.execute("DROP TABLE IF EXISTS temp.truth")

db.execute("""
    CREATE TEMP TABLE truth(
        s1 TEXT,
        cid TEXT,
        PRIMARY KEY(s1, cid)
    )
""")

for i in range(0, len(true_pairs), 100000):

    db.executemany(
        "INSERT OR IGNORE INTO truth VALUES (?, ?)",
        true_pairs[i:i + 100000]
    )

db.execute("CREATE INDEX idx_temp_truth_s1 ON truth(s1)")

# ---------------------------------------------------------
# Candidate recall
# ---------------------------------------------------------

found = db.execute("""
    SELECT COUNT(*)
    FROM truth t
    JOIN cand c
      ON c.s1 = t.s1
     AND c.cid = t.cid
""").fetchone()[0]

total_true = len(true_pairs)

recall = found / total_true if total_true else 0

candidate_precision = (
    found / candidate_count
    if candidate_count
    else 0
)

# ---------------------------------------------------------
# S1-level complete coverage
# ---------------------------------------------------------

fully_covered = db.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT
            t.s1,
            COUNT(*) AS true_count,
            SUM(
                CASE
                    WHEN c.cid IS NOT NULL THEN 1
                    ELSE 0
                END
            ) AS found_count
        FROM truth t
        LEFT JOIN cand c
          ON c.s1 = t.s1
         AND c.cid = t.cid
        GROUP BY t.s1
        HAVING true_count = found_count
    )
""").fetchone()[0]

s1_recall = (
    fully_covered / len(true_by_s1)
    if true_by_s1
    else 0
)

print()
print("=" * 65)
print("BLOCKING EVALUATION")
print("=" * 65)

print(f"Candidate pairs              : {candidate_count:,}")
print(f"True pairs                   : {total_true:,}")
print(f"True pairs found             : {found:,}")
print(f"Pair recall                  : {recall:.4%}")
print(f"Candidate precision          : {candidate_precision:.4%}")
print(
    f"S1 fully covered             : "
    f"{fully_covered:,}/{len(true_by_s1):,} "
    f"({s1_recall:.4%})"
)

# ---------------------------------------------------------
# Missed true-pair feature analysis
# ---------------------------------------------------------

print()
print("Analyzing missed true pairs...")

query = """
SELECT
    COUNT(*) AS missed,

    SUM(
        CASE WHEN
            s.name_key <> ''
            AND s.name_key = r.name_key
        THEN 1 ELSE 0 END
    ) AS name_exact,

    SUM(
        CASE WHEN
            s.sorted_name_key <> ''
            AND s.sorted_name_key = r.sorted_name_key
        THEN 1 ELSE 0 END
    ) AS sorted_name,

    SUM(
        CASE WHEN
            s.name_token_key <> ''
            AND s.name_token_key = r.name_token_key
        THEN 1 ELSE 0 END
    ) AS name_token,

    SUM(
        CASE WHEN
            s.first_name_token <> ''
            AND s.first_name_token = r.first_name_token
        THEN 1 ELSE 0 END
    ) AS first_name,

    SUM(
        CASE WHEN
            s.address_key <> ''
            AND s.address_key = r.address_key
        THEN 1 ELSE 0 END
    ) AS address_exact,

    SUM(
        CASE WHEN
            s.address_number <> ''
            AND s.address_number = r.address_number
        THEN 1 ELSE 0 END
    ) AS address_number,

    SUM(
        CASE WHEN
            s.name_prefix_sig <> ''
            AND s.name_prefix_sig = r.name_prefix_sig
        THEN 1 ELSE 0 END
    ) AS name_prefix,

    SUM(
        CASE WHEN
            s.name_suffix_sig <> ''
            AND s.name_suffix_sig = r.name_suffix_sig
        THEN 1 ELSE 0 END
    ) AS name_suffix,

    SUM(
        CASE WHEN
            s.address_token_sig <> ''
            AND s.address_token_sig = r.address_token_sig
        THEN 1 ELSE 0 END
    ) AS address_token

FROM truth t

JOIN s1 s
  ON s.entity_id = t.s1

JOIN records r
  ON r.entity_id = t.cid

LEFT JOIN cand c
  ON c.s1 = t.s1
 AND c.cid = t.cid

WHERE c.cid IS NULL
"""

row = db.execute(query).fetchone()

missed = row[0]

feature_names = [
    "name_key",
    "sorted_name_key",
    "name_token_key",
    "first_name_token",
    "address_key",
    "address_number",
    "name_prefix_sig",
    "name_suffix_sig",
    "address_token_sig",
]

feature_counts = row[1:]

for name, count in zip(feature_names, feature_counts):
    pct = count / missed if missed else 0
    print(f"{name:20s}: {count:8,} ({pct:6.2%})")

# ---------------------------------------------------------
# Any name/address signal
# ---------------------------------------------------------

any_name = db.execute("""
SELECT COUNT(*)
FROM truth t
JOIN s1 s ON s.entity_id = t.s1
JOIN records r ON r.entity_id = t.cid
LEFT JOIN cand c
  ON c.s1 = t.s1
 AND c.cid = t.cid
WHERE c.cid IS NULL
AND (
       (s.name_key <> '' AND s.name_key = r.name_key)
    OR (s.sorted_name_key <> '' AND s.sorted_name_key = r.sorted_name_key)
    OR (s.name_token_key <> '' AND s.name_token_key = r.name_token_key)
    OR (s.first_name_token <> '' AND s.first_name_token = r.first_name_token)
    OR (s.name_prefix_sig <> '' AND s.name_prefix_sig = r.name_prefix_sig)
    OR (s.name_suffix_sig <> '' AND s.name_suffix_sig = r.name_suffix_sig)
)
""").fetchone()[0]

any_address = db.execute("""
SELECT COUNT(*)
FROM truth t
JOIN s1 s ON s.entity_id = t.s1
JOIN records r ON r.entity_id = t.cid
LEFT JOIN cand c
  ON c.s1 = t.s1
 AND c.cid = t.cid
WHERE c.cid IS NULL
AND (
       (s.address_key <> '' AND s.address_key = r.address_key)
    OR (s.address_number <> '' AND s.address_number = r.address_number)
    OR (s.address_token_sig <> '' AND s.address_token_sig = r.address_token_sig)
)
""").fetchone()[0]

print()
print(f"Missed with ANY exact name signal    : "
      f"{any_name:,} ({any_name/missed:.2%})")

print(f"Missed with ANY exact address signal : "
      f"{any_address:,} ({any_address/missed:.2%})")

print()
print("=" * 65)
print(f"Runtime: {time.time() - start:.1f} seconds")
print("=" * 65)

db.close()