import sqlite3
import time

DB = r"results\blocking\blocking_train_full.db"

R4_CAP = 500
R5_CAP = 500
R6_CAP = 250

conn = sqlite3.connect(DB)

conn.execute("PRAGMA cache_size=-500000")
conn.execute("PRAGMA temp_store=MEMORY")
conn.execute("PRAGMA synchronous=NORMAL")

start = time.time()

def count_candidates():
    return conn.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]


print("=" * 80)
print("FULL V3 BLOCKING - EXACT R4/R5/R6")
print("=" * 80)

before = count_candidates()
print("Current candidates:", before)


# ============================================================
# R4
# country + address_number + address_token_sig
# Frequency cap <= 500
# ============================================================

print("\n" + "-" * 80)
print("R4: country + address_number + address_token_sig")
print("Frequency cap:", R4_CAP)
print("-" * 80)

t = time.time()

conn.execute("""
ATTACH DATABASE ? AS srcdb
""", (DB,))

conn.execute("""
INSERT OR IGNORE INTO candidates (
    source1_entity_id,
    candidate_entity_id
)
SELECT
    s.entity_id,
    r.entity_id
FROM s1 AS s
JOIN srcdb.records AS r
  ON r.country = s.country
 AND r.address_number = s.address_number
 AND r.address_token_sig = s.address_token_sig
JOIN (
    SELECT
        country,
        address_number,
        address_token_sig
    FROM srcdb.records
    WHERE address_number != ''
      AND address_token_sig != ''
    GROUP BY
        country,
        address_number,
        address_token_sig
    HAVING COUNT(*) <= 500
) AS k
  ON k.country = r.country
 AND k.address_number = r.address_number
 AND k.address_token_sig = r.address_token_sig
WHERE s.address_number != ''
  AND s.address_token_sig != ''
""")

conn.commit()

after_r4 = count_candidates()

print("R4 new candidates:", after_r4 - before)
print("R4 total:", after_r4)
print("R4 runtime:", round(time.time() - t, 2), "sec")


# ============================================================
# R5
# country + first_name_token + address_token_sig
# Frequency cap <= 500
# ============================================================

print("\n" + "-" * 80)
print("R5: country + first_name_token + address_token_sig")
print("Frequency cap:", R5_CAP)
print("-" * 80)

t = time.time()

before_r5 = after_r4

conn.execute("""
INSERT OR IGNORE INTO candidates (
    source1_entity_id,
    candidate_entity_id
)
SELECT
    s.entity_id,
    r.entity_id
FROM s1 AS s
JOIN srcdb.records AS r
  ON r.country = s.country
 AND r.first_name_token = s.first_name_token
 AND r.address_token_sig = s.address_token_sig
JOIN (
    SELECT
        country,
        first_name_token,
        address_token_sig
    FROM srcdb.records
    WHERE first_name_token != ''
      AND address_token_sig != ''
    GROUP BY
        country,
        first_name_token,
        address_token_sig
    HAVING COUNT(*) <= 500
) AS k
  ON k.country = r.country
 AND k.first_name_token = r.first_name_token
 AND k.address_token_sig = r.address_token_sig
WHERE s.first_name_token != ''
  AND s.address_token_sig != ''
""")

conn.commit()

after_r5 = count_candidates()

print("R5 new candidates:", after_r5 - before_r5)
print("R5 total:", after_r5)
print("R5 runtime:", round(time.time() - t, 2), "sec")


# ============================================================
# R6
# country + name_prefix_sig + address_number
# Frequency cap <= 250
# ============================================================

print("\n" + "-" * 80)
print("R6: country + name_prefix_sig + address_number")
print("Frequency cap:", R6_CAP)
print("-" * 80)

t = time.time()

before_r6 = after_r5

conn.execute("""
INSERT OR IGNORE INTO candidates (
    source1_entity_id,
    candidate_entity_id
)
SELECT
    s.entity_id,
    r.entity_id
FROM s1 AS s
JOIN srcdb.records AS r
  ON r.country = s.country
 AND r.name_prefix_sig = s.name_prefix_sig
 AND r.address_number = s.address_number
JOIN (
    SELECT
        country,
        name_prefix_sig,
        address_number
    FROM srcdb.records
    WHERE name_prefix_sig != ''
      AND address_number != ''
    GROUP BY
        country,
        name_prefix_sig,
        address_number
    HAVING COUNT(*) <= 250
) AS k
  ON k.country = r.country
 AND k.name_prefix_sig = r.name_prefix_sig
 AND k.address_number = r.address_number
WHERE s.name_prefix_sig != ''
  AND s.address_number != ''
""")

conn.commit()

after_r6 = count_candidates()

print("R6 new candidates:", after_r6 - before_r6)
print("R6 total:", after_r6)
print("R6 runtime:", round(time.time() - t, 2), "sec")


# ============================================================
# FINAL
# ============================================================

s1_count = conn.execute(
    "SELECT COUNT(*) FROM s1"
).fetchone()[0]

candidate_s1 = conn.execute(
    "SELECT COUNT(DISTINCT source1_entity_id) FROM candidates"
).fetchone()[0]

final_count = count_candidates()

print("\n" + "=" * 80)
print("FULL V3 BLOCKING COMPLETE")
print("=" * 80)

print("S1 entities:", f"{s1_count:,}")
print("R1-R3 base:", f"{before:,}")
print("R4 added:", f"{after_r4 - before:,}")
print("R5 added:", f"{after_r5 - after_r4:,}")
print("R6 added:", f"{after_r6 - after_r5:,}")
print("FINAL candidates:", f"{final_count:,}")
print("S1 with candidates:", f"{candidate_s1:,}")
print("S1 coverage:", f"{candidate_s1 / s1_count * 100:.2f}%")
print("Average candidates/S1:", f"{final_count / s1_count:.2f}")
print("Total runtime:", f"{(time.time() - start) / 60:.2f} min")

conn.close()
