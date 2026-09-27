import sqlite3
import time

DB = r"results\blocking\blocking_train_full.db"

conn = sqlite3.connect(DB)
conn.execute("PRAGMA cache_size=-500000")
conn.execute("PRAGMA temp_store=MEMORY")
conn.execute("PRAGMA synchronous=NORMAL")

def run_rule(name, sql):
    before = conn.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    print("\n" + "=" * 70)
    print(name)
    print("Before:", before)

    start = time.time()

    conn.execute(sql)
    conn.commit()

    after = conn.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    print("New candidates:", after - before)
    print("Total candidates:", after)
    print("Runtime:", round((time.time() - start) / 60, 2), "min")


# R4:
# country + address_number + address_token_sig
# Keep at most 500 candidates per S1.
run_rule("R4: address_number + address_token (cap 500)", """
INSERT OR IGNORE INTO candidates
SELECT source1_entity_id, candidate_entity_id
FROM (
    SELECT
        s.entity_id AS source1_entity_id,
        r.entity_id AS candidate_entity_id,
        ROW_NUMBER() OVER (
            PARTITION BY s.entity_id
            ORDER BY r.entity_id
        ) AS rn
    FROM s1 s
    JOIN records r
      ON r.country = s.country
     AND r.address_number = s.address_number
     AND r.address_token_sig = s.address_token_sig
    WHERE s.address_number <> ''
      AND s.address_token_sig <> ''
)
WHERE rn <= 500
""")


# R5:
# country + first_name_token + address_token_sig
# Keep at most 500 candidates per S1.
run_rule("R5: first_name_token + address_token (cap 500)", """
INSERT OR IGNORE INTO candidates
SELECT source1_entity_id, candidate_entity_id
FROM (
    SELECT
        s.entity_id AS source1_entity_id,
        r.entity_id AS candidate_entity_id,
        ROW_NUMBER() OVER (
            PARTITION BY s.entity_id
            ORDER BY r.entity_id
        ) AS rn
    FROM s1 s
    JOIN records r
      ON r.country = s.country
     AND r.first_name_token = s.first_name_token
     AND r.address_token_sig = s.address_token_sig
    WHERE s.first_name_token <> ''
      AND s.address_token_sig <> ''
)
WHERE rn <= 500
""")


total = conn.execute(
    "SELECT COUNT(*) FROM candidates"
).fetchone()[0]

covered = conn.execute(
    "SELECT COUNT(DISTINCT source1_entity_id) FROM candidates"
).fetchone()[0]

s1_count = conn.execute(
    "SELECT COUNT(*) FROM s1"
).fetchone()[0]

print("\n" + "=" * 70)
print("FULL V3 BASE + R4 + R5 COMPLETE")
print("=" * 70)

print("S1:", s1_count)
print("Candidates:", total)
print("S1 with candidates:", covered)
print("Coverage:", round(covered / s1_count * 100, 2), "%")
print("Average candidates / S1:", round(total / s1_count, 2))

conn.close()
