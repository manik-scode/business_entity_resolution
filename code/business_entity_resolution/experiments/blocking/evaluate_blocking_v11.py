import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import sqlite3
import csv
import time


ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "results" / "blocking" / "blocking_train_v2.db"
GT_PATH = (
    ROOT / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"
    if (ROOT / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv").exists()
    else ROOT / "dataset" / "train" / "ground_truth.tsv"
)

SAMPLE_SIZE = 1000

# Caps to test
CAPS = [5, 10, 25, 50, 100, 250, 500, 1000, 2000]


# ============================================================
# LOAD SAMPLE + GROUND TRUTH
# ============================================================

def load_sample(conn):
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS temp.v11_sample_s1")
    cur.execute("DROP TABLE IF EXISTS temp.v11_gt")

    cur.execute("""
        CREATE TEMP TABLE v11_sample_s1 (
            entity_id TEXT PRIMARY KEY
        )
    """)

    cur.execute("""
        INSERT INTO v11_sample_s1(entity_id)
        SELECT entity_id FROM s1 ORDER BY rowid LIMIT ?
    """, (SAMPLE_SIZE,))

    cur.execute("""
        CREATE TEMP TABLE v11_gt (
            source1_entity_id TEXT NOT NULL,
            matched_entity_id TEXT NOT NULL,
            PRIMARY KEY(source1_entity_id, matched_entity_id)
        )
    """)

    sample_set = set(
        row[0] for row in cur.execute("SELECT entity_id FROM v11_sample_s1").fetchall()
    )

    count = len(sample_set)
    matched_s1_seen = set()

    with open(GT_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            s1_id = row["source1_entity_id"]

            if s1_id not in sample_set:
                continue

            matched_s1_seen.add(s1_id)
            matched = row.get("matched_entity_ids", "")

            if matched:
                for candidate_id in matched.split(","):
                    candidate_id = candidate_id.strip()

                    if candidate_id:
                        cur.execute(
                            """
                            INSERT OR IGNORE INTO v11_gt
                            (source1_entity_id, matched_entity_id)
                            VALUES (?, ?)
                            """,
                            (s1_id, candidate_id),
                        )

            if len(matched_s1_seen) == len(sample_set):
                break

    conn.commit()

    true_pairs = cur.execute(
        "SELECT COUNT(*) FROM v11_gt"
    ).fetchone()[0]

    print(f"S1 sample       : {count:,}")
    print(f"True pairs      : {true_pairs:,}")

def build_v9(conn):

    cur = conn.cursor()

    cur.execute("""
        CREATE TEMP TABLE v11_s1 AS
        SELECT s.*
        FROM s1 s
        INNER JOIN v11_sample_s1 x
            ON s.entity_id = x.entity_id
    """)

    cur.execute("""
        CREATE TEMP TABLE v11_candidates (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,
            PRIMARY KEY(source1_entity_id, candidate_entity_id)
        )
    """)

    cur.execute("CREATE INDEX idx_v11_s1_entity ON v11_s1(entity_id)")
    cur.execute("CREATE INDEX idx_v11_s1_country_name ON v11_s1(country, name_key)")
    cur.execute("CREATE INDEX idx_v11_s1_country_sorted ON v11_s1(country, sorted_name_key)")
    cur.execute("CREATE INDEX idx_v11_s1_token_number ON v11_s1(country, name_token_key, address_number)")
    cur.execute("CREATE INDEX idx_v11_s1_first_number ON v11_s1(country, first_name_token, address_number)")
    cur.execute("CREATE INDEX idx_v11_s1_first_prefix ON v11_s1(country, first_name_token, name_prefix_sig)")

    # Precalculate R18 accepted keys (cap 500)
    cur.execute("DROP TABLE IF EXISTS temp.v11_keys")
    cur.execute("""
        CREATE TEMP TABLE v11_keys AS
        SELECT DISTINCT
            country,
            address_number,
            address_token_sig
        FROM v11_s1
        WHERE address_number <> ''
          AND address_token_sig <> ''
    """)
    cur.execute("""
        CREATE UNIQUE INDEX idx_v11_keys
        ON v11_keys(country, address_number, address_token_sig)
    """)

    cur.execute("DROP TABLE IF EXISTS temp.v11_r18_freq")
    cur.execute("""
        CREATE TEMP TABLE v11_r18_freq AS
        SELECT
            k.country,
            k.address_number,
            k.address_token_sig,
            COUNT(*) AS freq
        FROM v11_keys k
        CROSS JOIN records r
          ON r.country = k.country
         AND r.address_token_sig = k.address_token_sig
         AND r.address_number = k.address_number
        GROUP BY
            k.country,
            k.address_number,
            k.address_token_sig
        HAVING COUNT(*) <= 500
    """)
    cur.execute("""
        CREATE UNIQUE INDEX idx_v11_r18_freq
        ON v11_r18_freq(country, address_number, address_token_sig)
    """)

    conn.commit()

    rules = [

        (
            "R1 exact name",
            """
            INSERT OR IGNORE INTO v11_candidates
            SELECT s.entity_id, r.entity_id
            FROM v11_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_key = s.name_key
            WHERE s.name_key <> ''
              AND r.name_key <> ''
            """
        ),

        (
            "R2 sorted name",
            """
            INSERT OR IGNORE INTO v11_candidates
            SELECT s.entity_id, r.entity_id
            FROM v11_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.sorted_name_key = s.sorted_name_key
            WHERE s.sorted_name_key <> ''
              AND r.sorted_name_key <> ''
            """
        ),

        (
            "R3 name token + number",
            """
            INSERT OR IGNORE INTO v11_candidates
            SELECT s.entity_id, r.entity_id
            FROM v11_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_token_key = s.name_token_key
             AND r.address_number = s.address_number
            WHERE s.name_token_key <> ''
              AND r.name_token_key <> ''
              AND s.address_number <> ''
              AND r.address_number <> ''
            """
        ),

        (
            "R4 first token + number",
            """
            INSERT OR IGNORE INTO v11_candidates
            SELECT s.entity_id, r.entity_id
            FROM v11_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.first_name_token = s.first_name_token
             AND r.address_number = s.address_number
            WHERE s.first_name_token <> ''
              AND r.first_name_token <> ''
              AND s.address_number <> ''
              AND r.address_number <> ''
            """
        ),

        (
            "R5 first token + number + prefix",
            """
            INSERT OR IGNORE INTO v11_candidates
            SELECT s.entity_id, r.entity_id
            FROM v11_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.first_name_token = s.first_name_token
             AND r.address_number = s.address_number
             AND r.name_prefix_sig = s.name_prefix_sig
            WHERE s.first_name_token <> ''
              AND r.first_name_token <> ''
              AND s.address_number <> ''
              AND r.address_number <> ''
              AND s.name_prefix_sig <> ''
              AND r.name_prefix_sig <> ''
            """
        ),

        (
            "R6 first token + number + suffix",
            """
            INSERT OR IGNORE INTO v11_candidates
            SELECT s.entity_id, r.entity_id
            FROM v11_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.first_name_token = s.first_name_token
             AND r.address_number = s.address_number
             AND r.name_suffix_sig = s.name_suffix_sig
            WHERE s.first_name_token <> ''
              AND r.first_name_token <> ''
              AND s.address_number <> ''
              AND r.address_number <> ''
              AND s.name_suffix_sig <> ''
              AND r.name_suffix_sig <> ''
            """
        ),

        (
            "R7 exact address + first token",
            """
            INSERT OR IGNORE INTO v11_candidates
            SELECT s.entity_id, r.entity_id
            FROM v11_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.address_key = s.address_key
             AND r.first_name_token = s.first_name_token
            WHERE s.address_key <> ''
              AND r.address_key <> ''
              AND s.first_name_token <> ''
              AND r.first_name_token <> ''
            """
        ),

        (
            "R8 prefix + address token",
            """
            INSERT OR IGNORE INTO v11_candidates
            SELECT s.entity_id, r.entity_id
            FROM v11_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_prefix_sig = s.name_prefix_sig
             AND r.address_token_sig = s.address_token_sig
            WHERE s.name_prefix_sig <> ''
              AND r.name_prefix_sig <> ''
              AND s.address_token_sig <> ''
              AND r.address_token_sig <> ''
            """
        ),

        (
            "R9 suffix + address token",
            """
            INSERT OR IGNORE INTO v11_candidates
            SELECT s.entity_id, r.entity_id
            FROM v11_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_suffix_sig = s.name_suffix_sig
             AND r.address_token_sig = s.address_token_sig
            WHERE s.name_suffix_sig <> ''
              AND r.name_suffix_sig <> ''
              AND s.address_token_sig <> ''
              AND r.address_token_sig <> ''
            """
        ),

        (
            "R10 first token + address token",
            """
            INSERT OR IGNORE INTO v11_candidates
            SELECT s.entity_id, r.entity_id
            FROM v11_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.first_name_token = s.first_name_token
             AND r.address_token_sig = s.address_token_sig
            WHERE s.first_name_token <> ''
              AND r.first_name_token <> ''
              AND s.address_token_sig <> ''
              AND r.address_token_sig <> ''
            """
        ),

        (
            "R17 exact address",
            """
            INSERT OR IGNORE INTO v11_candidates
            SELECT s.entity_id, r.entity_id
            FROM v11_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.address_key = s.address_key
            WHERE s.address_key <> ''
              AND r.address_key <> ''
            """
        ),

        (
            "R18 address number + token cap 500",
            """
            INSERT OR IGNORE INTO v11_candidates
            SELECT s.entity_id, r.entity_id
            FROM v11_s1 s
            JOIN v11_r18_freq f
              ON f.country = s.country
             AND f.address_number = s.address_number
             AND f.address_token_sig = s.address_token_sig
            JOIN records r
              ON r.country = s.country
             AND r.address_number = s.address_number
             AND r.address_token_sig = s.address_token_sig
            WHERE s.address_number <> ''
              AND s.address_token_sig <> ''
            """
        ),

        (
            "R19 prefix + number",
            """
            INSERT OR IGNORE INTO v11_candidates
            SELECT s.entity_id, r.entity_id
            FROM v11_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_prefix_sig = s.name_prefix_sig
             AND r.address_number = s.address_number
            WHERE s.name_prefix_sig <> ''
              AND r.name_prefix_sig <> ''
              AND s.address_number <> ''
              AND r.address_number <> ''
            """
        ),

        (
            "R20 suffix + number",
            """
            INSERT OR IGNORE INTO v11_candidates
            SELECT s.entity_id, r.entity_id
            FROM v11_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_suffix_sig = s.name_suffix_sig
             AND r.address_number = s.address_number
            WHERE s.name_suffix_sig <> ''
              AND r.name_suffix_sig <> ''
              AND s.address_number <> ''
              AND r.address_number <> ''
            """
        ),
    ]

    print("\nRebuilding V9...")

    for name, sql in rules:
        start = time.time()

        before = cur.execute(
            "SELECT COUNT(*) FROM v11_candidates"
        ).fetchone()[0]

        cur.execute(sql)
        conn.commit()

        after = cur.execute(
            "SELECT COUNT(*) FROM v11_candidates"
        ).fetchone()[0]

        print(
            f"{name:<40} "
            f"+{after-before:>9,} "
            f"total={after:>9,} "
            f"{time.time()-start:.2f}s"
        )

    return cur.execute(
        "SELECT COUNT(*) FROM v11_candidates"
    ).fetchone()[0]


# ============================================================
# CREATE MISSED PAIRS
# ============================================================

def create_missed(conn):

    cur = conn.cursor()

    cur.execute("""
        CREATE TEMP TABLE v11_missed AS
        SELECT
            g.source1_entity_id,
            g.matched_entity_id
        FROM v11_gt g
        LEFT JOIN v11_candidates c
          ON c.source1_entity_id = g.source1_entity_id
         AND c.candidate_entity_id = g.matched_entity_id
        WHERE c.candidate_entity_id IS NULL
    """)

    conn.commit()

    missed = cur.execute(
        "SELECT COUNT(*) FROM v11_missed"
    ).fetchone()[0]

    print(f"\nV9 missed pairs: {missed:,}")

    return missed


# ============================================================
# FREQUENCY RECOVERY ANALYSIS
# ============================================================

def analyse_feature(conn, feature, label):

    cur = conn.cursor()

    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    # How many missed pairs have identical feature?
    total_signal = cur.execute(f"""
        SELECT COUNT(*)
        FROM v11_missed m
        JOIN s1 s
          ON s.entity_id = m.source1_entity_id
        JOIN records r
          ON r.entity_id = m.matched_entity_id
        WHERE s.{feature} <> ''
          AND r.{feature} <> ''
          AND s.{feature} = r.{feature}
    """).fetchone()[0]

    print(
        f"Missed pairs sharing {feature}: "
        f"{total_signal:,}"
    )

    print("\nRecovery by frequency cap:")

    print(
        f"{'CAP':>8} "
        f"{'RECOVERED':>12} "
        f"{'TOTAL SIGNAL':>14} "
        f"{'RECOVERY %':>12}"
    )

    cur.execute("DROP TABLE IF EXISTS temp.v11_feat_freq")
    cur.execute(f"""
        CREATE TEMP TABLE v11_feat_freq AS
        SELECT
            m.source1_entity_id,
            m.matched_entity_id,
            (
                SELECT COUNT(*)
                FROM records rr
                WHERE rr.country = r.country
                  AND rr.{feature} = r.{feature}
            ) AS freq
        FROM v11_missed m
        JOIN s1 s
          ON s.entity_id = m.source1_entity_id
        JOIN records r
          ON r.entity_id = m.matched_entity_id
        WHERE s.{feature} <> ''
          AND r.{feature} <> ''
          AND s.{feature} = r.{feature}
    """)
    conn.commit()

    for cap in CAPS:

        recovered = cur.execute(
            "SELECT COUNT(*) FROM v11_feat_freq WHERE freq <= ?",
            (cap,)
        ).fetchone()[0]

        pct = (
            recovered / total_signal * 100
            if total_signal
            else 0
        )

        print(
            f"{cap:>8} "
            f"{recovered:>12,} "
            f"{total_signal:>14,} "
            f"{pct:>11.2f}%"
        )


# ============================================================
# COMBINED SELECTIVE RULE ANALYSIS
# ============================================================

def analyse_combinations(conn):

    cur = conn.cursor()

    print("\n" + "=" * 80)
    print("COMBINATION RECOVERY ANALYSIS")
    print("=" * 80)

    combinations = [
        (
            "first_name_token + address_token_sig",
            "s.first_name_token = r.first_name_token "
            "AND s.address_token_sig = r.address_token_sig "
            "AND s.first_name_token <> '' "
            "AND s.address_token_sig <> ''"
        ),
        (
            "name_prefix_sig + address_token_sig",
            "s.name_prefix_sig = r.name_prefix_sig "
            "AND s.address_token_sig = r.address_token_sig "
            "AND s.name_prefix_sig <> '' "
            "AND s.address_token_sig <> ''"
        ),
        (
            "name_suffix_sig + address_token_sig",
            "s.name_suffix_sig = r.name_suffix_sig "
            "AND s.address_token_sig = r.address_token_sig "
            "AND s.name_suffix_sig <> '' "
            "AND s.address_token_sig <> ''"
        ),
        (
            "first_name_token + address_number",
            "s.first_name_token = r.first_name_token "
            "AND s.address_number = r.address_number "
            "AND s.first_name_token <> '' "
            "AND s.address_number <> ''"
        ),
        (
            "name_prefix_sig + address_number",
            "s.name_prefix_sig = r.name_prefix_sig "
            "AND s.address_number = r.address_number "
            "AND s.name_prefix_sig <> '' "
            "AND s.address_number <> ''"
        ),
        (
            "name_suffix_sig + address_number",
            "s.name_suffix_sig = r.name_suffix_sig "
            "AND s.address_number = r.address_number "
            "AND s.name_suffix_sig <> '' "
            "AND s.address_number <> ''"
        ),
    ]

    for label, condition in combinations:

        count = cur.execute(f"""
            SELECT COUNT(*)
            FROM v11_missed m
            JOIN s1 s
              ON s.entity_id = m.source1_entity_id
            JOIN records r
              ON r.entity_id = m.matched_entity_id
            WHERE {condition}
        """).fetchone()[0]

        print(f"{label:<45} {count:>7,}")


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    print("=" * 80)
    print("V11 FINAL BLOCKING FORENSIC ANALYSIS")
    print("=" * 80)

    print(f"\nDB: {DB_PATH}")
    print(f"GT: {GT_PATH}")
    print(f"Sample: {SAMPLE_SIZE}")

    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    if not GT_PATH.exists():
        raise FileNotFoundError(GT_PATH)

    conn = sqlite3.connect(
        str(DB_PATH),
        timeout=120
    )

    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -100000")

    try:

        load_sample(conn)

        total_candidates = build_v9(conn)

        missed = create_missed(conn)

        analyse_feature(
            conn,
            "address_token_sig",
            "ADDRESS TOKEN FREQUENCY"
        )

        analyse_feature(
            conn,
            "first_name_token",
            "FIRST NAME TOKEN FREQUENCY"
        )

        analyse_feature(
            conn,
            "name_prefix_sig",
            "NAME PREFIX FREQUENCY"
        )

        analyse_feature(
            conn,
            "name_suffix_sig",
            "NAME SUFFIX FREQUENCY"
        )

        analyse_combinations(conn)

        print("\n" + "=" * 80)
        print("V11 COMPLETE")
        print("=" * 80)

        print(f"V9 candidates : {total_candidates:,}")
        print(f"V9 missed     : {missed:,}")
        print(
            f"Runtime       : "
            f"{(time.time() - start) / 60:.2f} minutes"
        )

        print(
            "\nNo new blocking rule was added."
            "\nThis is forensic analysis only."
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
