from pathlib import Path
import sqlite3
import csv
import sys
import time
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "results" / "blocking" / "blocking_train_v2.db"
GT_PATH = (
    ROOT / "dataset" / "train" / "ground_truth.tsv"
    if (ROOT / "dataset" / "train" / "ground_truth.tsv").exists()
    else ROOT / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"
)

SAMPLE_SIZE = 1000

# V9 optimized cap
R18_CAP = 500


# ============================================================
# HELPERS
# ============================================================

def qident(name: str) -> str:
    """Safely quote SQLite identifiers."""
    return '"' + name.replace('"', '""') + '"'


def percentile(values, p):
    if not values:
        return 0

    values = sorted(values)

    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values))

    if f == c:
        return values[f]

    return values[f] + (values[c] - values[f]) * (k - f)


def print_rule_header(name):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(conn, limit=SAMPLE_SIZE):

    print("\nLoading ground truth...")

    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS temp.v10_sample_s1")
    cur.execute("DROP TABLE IF EXISTS temp.v10_gt")

    cur.execute("""
        CREATE TEMP TABLE v10_sample_s1 (
            source1_entity_id TEXT PRIMARY KEY
        )
    """)

    cur.execute("""
        INSERT INTO v10_sample_s1 (source1_entity_id)
        SELECT entity_id FROM s1 ORDER BY rowid LIMIT ?
    """, (limit,))

    cur.execute("""
        CREATE TEMP TABLE v10_gt (
            source1_entity_id TEXT NOT NULL,
            matched_entity_id TEXT NOT NULL,
            PRIMARY KEY(source1_entity_id, matched_entity_id)
        )
    """)

    sample_set = set(
        row[0] for row in cur.execute("SELECT source1_entity_id FROM v10_sample_s1").fetchall()
    )

    total_rows = len(sample_set)
    true_pair_count = 0
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
                ids = [
                    x.strip()
                    for x in matched.split(",")
                    if x.strip()
                ]

                for candidate_id in ids:

                    cur.execute(
                        """
                        INSERT OR IGNORE INTO v10_gt
                        (source1_entity_id, matched_entity_id)
                        VALUES (?, ?)
                        """,
                        (s1_id, candidate_id),
                    )

                    true_pair_count += 1

            if len(matched_s1_seen) == len(sample_set):
                break

    conn.commit()

    print(f"S1 sample rows       : {total_rows:,}")
    print(f"True matched pairs   : {true_pair_count:,}")

    return total_rows, true_pair_count


# ============================================================
# BUILD V9 CANDIDATES
# ============================================================

def build_v9_candidates(conn):

    print("\n" + "=" * 80)
    print("BUILDING V9 OPTIMIZED CANDIDATE SET")
    print("=" * 80)

    cur = conn.cursor()

    cur.execute("""
        CREATE TEMP TABLE v10_s1 AS
        SELECT r.*
        FROM records r
        INNER JOIN v10_sample_s1 s
            ON r.entity_id = s.source1_entity_id
        WHERE 1 = 0
    """)

    # The S1 rows are stored in the permanent s1 table.
    # Use that table instead.
    cur.execute("DROP TABLE v10_s1")

    cur.execute("""
        CREATE TEMP TABLE v10_s1 AS
        SELECT s.*
        FROM s1 s
        INNER JOIN v10_sample_s1 sample
            ON s.entity_id = sample.source1_entity_id
    """)

    cur.execute("""
        CREATE INDEX idx_v10_s1_entity
        ON v10_s1(entity_id)
    """)

    cur.execute("""
        CREATE INDEX idx_v10_s1_country_name
        ON v10_s1(country, name_key)
    """)

    cur.execute("""
        CREATE INDEX idx_v10_s1_country_sorted
        ON v10_s1(country, sorted_name_key)
    """)

    cur.execute("""
        CREATE INDEX idx_v10_s1_token_number
        ON v10_s1(country, name_token_key, address_number)
    """)

    cur.execute("""
        CREATE INDEX idx_v10_s1_first_number
        ON v10_s1(country, first_name_token, address_number)
    """)

    cur.execute("""
        CREATE INDEX idx_v10_s1_first_prefix
        ON v10_s1(country, first_name_token, name_prefix_sig)
    """)

    cur.execute("""
        CREATE TEMP TABLE v10_candidates (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,
            PRIMARY KEY(source1_entity_id, candidate_entity_id)
        )
    """)

    cur.execute("""
        CREATE INDEX idx_v10_candidates_s1
        ON v10_candidates(source1_entity_id)
    """)

    conn.commit()

    def run_rule(rule_name, sql):

        start = time.time()

        before = cur.execute(
            "SELECT COUNT(*) FROM v10_candidates"
        ).fetchone()[0]

        cur.execute(sql)

        conn.commit()

        after = cur.execute(
            "SELECT COUNT(*) FROM v10_candidates"
        ).fetchone()[0]

        added = after - before

        print(
            f"{rule_name:<42} "
            f"+{added:>10,}   "
            f"total={after:>10,}   "
            f"time={time.time() - start:>7.2f}s"
        )

    # --------------------------------------------------------
    # V9 R1
    # Exact normalized name
    # --------------------------------------------------------

    run_rule(
        "R1 exact name",
        """
        INSERT OR IGNORE INTO v10_candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM v10_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.name_key = s.name_key
        WHERE s.name_key <> ''
          AND r.name_key <> ''
        """
    )

    # --------------------------------------------------------
    # V9 R2
    # Sorted normalized name
    # --------------------------------------------------------

    run_rule(
        "R2 sorted name",
        """
        INSERT OR IGNORE INTO v10_candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM v10_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.sorted_name_key = s.sorted_name_key
        WHERE s.sorted_name_key <> ''
          AND r.sorted_name_key <> ''
        """
    )

    # --------------------------------------------------------
    # V9 R3
    # Name token + address number
    # --------------------------------------------------------

    run_rule(
        "R3 name token + number",
        """
        INSERT OR IGNORE INTO v10_candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM v10_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.name_token_key = s.name_token_key
         AND r.address_number = s.address_number
        WHERE s.name_token_key <> ''
          AND r.name_token_key <> ''
          AND s.address_number <> ''
          AND r.address_number <> ''
        """
    )

    # --------------------------------------------------------
    # V9 R4
    # First name token + address number
    # --------------------------------------------------------

    run_rule(
        "R4 first token + number",
        """
        INSERT OR IGNORE INTO v10_candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM v10_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.first_name_token = s.first_name_token
         AND r.address_number = s.address_number
        WHERE s.first_name_token <> ''
          AND r.first_name_token <> ''
          AND s.address_number <> ''
          AND r.address_number <> ''
        """
    )

    # --------------------------------------------------------
    # V9 R5
    # First token + number + prefix
    # --------------------------------------------------------

    run_rule(
        "R5 first token + number + prefix",
        """
        INSERT OR IGNORE INTO v10_candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM v10_s1 s
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
    )

    # --------------------------------------------------------
    # V9 R6
    # First token + number + suffix
    # --------------------------------------------------------

    run_rule(
        "R6 first token + number + suffix",
        """
        INSERT OR IGNORE INTO v10_candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM v10_s1 s
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
    )

    # --------------------------------------------------------
    # V9 R7
    # Exact address + first token
    # --------------------------------------------------------

    run_rule(
        "R7 exact address + first token",
        """
        INSERT OR IGNORE INTO v10_candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM v10_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.address_key = s.address_key
         AND r.first_name_token = s.first_name_token
        WHERE s.address_key <> ''
          AND r.address_key <> ''
          AND s.first_name_token <> ''
          AND r.first_name_token <> ''
        """
    )

    # --------------------------------------------------------
    # V9 R8
    # Prefix + address token
    # --------------------------------------------------------

    run_rule(
        "R8 prefix + address token",
        """
        INSERT OR IGNORE INTO v10_candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM v10_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.name_prefix_sig = s.name_prefix_sig
         AND r.address_token_sig = s.address_token_sig
        WHERE s.name_prefix_sig <> ''
          AND r.name_prefix_sig <> ''
          AND s.address_token_sig <> ''
          AND r.address_token_sig <> ''
        """
    )

    # --------------------------------------------------------
    # V9 R9
    # Suffix + address token
    # --------------------------------------------------------

    run_rule(
        "R9 suffix + address token",
        """
        INSERT OR IGNORE INTO v10_candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM v10_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.name_suffix_sig = s.name_suffix_sig
         AND r.address_token_sig = s.address_token_sig
        WHERE s.name_suffix_sig <> ''
          AND r.name_suffix_sig <> ''
          AND s.address_token_sig <> ''
          AND r.address_token_sig <> ''
        """
    )

    # --------------------------------------------------------
    # V9 R10
    # First token + address token
    # --------------------------------------------------------

    run_rule(
        "R10 first token + address token",
        """
        INSERT OR IGNORE INTO v10_candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM v10_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.first_name_token = s.first_name_token
         AND r.address_token_sig = s.address_token_sig
        WHERE s.first_name_token <> ''
          AND r.first_name_token <> ''
          AND s.address_token_sig <> ''
          AND r.address_token_sig <> ''
        """
    )

    # --------------------------------------------------------
    # V9 R17
    # Exact address
    # --------------------------------------------------------

    run_rule(
        "R17 exact address",
        """
        INSERT OR IGNORE INTO v10_candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM v10_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.address_key = s.address_key
        WHERE s.address_key <> ''
          AND r.address_key <> ''
        """
    )

    # --------------------------------------------------------
    # V9 R18
    # Address number + address token
    #
    # IMPORTANT:
    # Frequency cap = 500
    # --------------------------------------------------------

    print("\nCalculating R18 key frequencies...")

    cur.execute("DROP TABLE IF EXISTS temp.v10_keys")
    cur.execute("""
        CREATE TEMP TABLE v10_keys AS
        SELECT DISTINCT
            country,
            address_number,
            address_token_sig
        FROM v10_s1
        WHERE address_number <> ''
          AND address_token_sig <> ''
    """)
    cur.execute("""
        CREATE UNIQUE INDEX idx_v10_keys
        ON v10_keys(country, address_number, address_token_sig)
    """)

    cur.execute("DROP TABLE IF EXISTS temp.v10_r18_freq")
    cur.execute("""
        CREATE TEMP TABLE v10_r18_freq AS
        SELECT
            k.country,
            k.address_number,
            k.address_token_sig,
            COUNT(*) AS freq
        FROM v10_keys k
        CROSS JOIN records r
          ON r.country = k.country
         AND r.address_token_sig = k.address_token_sig
         AND r.address_number = k.address_number
        GROUP BY
            k.country,
            k.address_number,
            k.address_token_sig
        HAVING COUNT(*) <= ?
    """, (R18_CAP,))

    cur.execute("""
        CREATE UNIQUE INDEX idx_v10_r18_freq
        ON v10_r18_freq(country, address_number, address_token_sig)
    """)

    conn.commit()

    r18_keys = cur.execute(
        "SELECT COUNT(*) FROM v10_r18_freq"
    ).fetchone()[0]

    print(f"R18 accepted keys (freq <= {R18_CAP}): {r18_keys:,}")

    run_rule(
        f"R18 number + address token (cap {R18_CAP})",
        """
        INSERT OR IGNORE INTO v10_candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM v10_s1 s
        JOIN v10_r18_freq f
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
    )

    # --------------------------------------------------------
    # V9 R19
    # Prefix + address number
    # --------------------------------------------------------

    run_rule(
        "R19 prefix + number",
        """
        INSERT OR IGNORE INTO v10_candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM v10_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.name_prefix_sig = s.name_prefix_sig
         AND r.address_number = s.address_number
        WHERE s.name_prefix_sig <> ''
          AND r.name_prefix_sig <> ''
          AND s.address_number <> ''
          AND r.address_number <> ''
        """
    )

    # --------------------------------------------------------
    # V9 R20
    # Suffix + address number
    # --------------------------------------------------------

    run_rule(
        "R20 suffix + number",
        """
        INSERT OR IGNORE INTO v10_candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM v10_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.name_suffix_sig = s.name_suffix_sig
         AND r.address_number = s.address_number
        WHERE s.name_suffix_sig <> ''
          AND r.name_suffix_sig <> ''
          AND s.address_number <> ''
          AND r.address_number <> ''
        """
    )

    total_candidates = cur.execute(
        "SELECT COUNT(*) FROM v10_candidates"
    ).fetchone()[0]

    print("\nV9 candidate generation complete.")
    print(f"Total candidates: {total_candidates:,}")

    return total_candidates


# ============================================================
# V9 RECALL
# ============================================================

def calculate_v9_recall(conn):

    cur = conn.cursor()

    true_pairs = cur.execute(
        "SELECT COUNT(*) FROM v10_gt"
    ).fetchone()[0]

    retrieved_pairs = cur.execute("""
        SELECT COUNT(*)
        FROM v10_gt g
        INNER JOIN v10_candidates c
          ON c.source1_entity_id = g.source1_entity_id
         AND c.candidate_entity_id = g.matched_entity_id
    """).fetchone()[0]

    missed_pairs = true_pairs - retrieved_pairs

    candidate_count = cur.execute(
        "SELECT COUNT(*) FROM v10_candidates"
    ).fetchone()[0]

    pair_recall = (
        retrieved_pairs / true_pairs * 100
        if true_pairs
        else 0
    )

    pair_precision = (
        retrieved_pairs / candidate_count * 100
        if candidate_count
        else 0
    )

    print("\n" + "=" * 80)
    print("V9 RECALL SUMMARY")
    print("=" * 80)

    print(f"True pairs              : {true_pairs:,}")
    print(f"Retrieved true pairs    : {retrieved_pairs:,}")
    print(f"Missed true pairs       : {missed_pairs:,}")
    print(f"Candidate pairs         : {candidate_count:,}")
    print(f"Candidate pair recall   : {pair_recall:.4f}%")
    print(f"Candidate pair precision : {pair_precision:.4f}%")

    # Entity-level coverage
    total_entities = cur.execute("""
        SELECT COUNT(DISTINCT source1_entity_id)
        FROM v10_gt
    """).fetchone()[0]

    entities_with_all_matches = cur.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT
                g.source1_entity_id
            FROM v10_gt g
            LEFT JOIN v10_candidates c
              ON c.source1_entity_id = g.source1_entity_id
             AND c.candidate_entity_id = g.matched_entity_id
            GROUP BY g.source1_entity_id
            HAVING COUNT(*) = SUM(
                CASE
                    WHEN c.candidate_entity_id IS NOT NULL
                    THEN 1
                    ELSE 0
                END
            )
        )
    """).fetchone()[0]

    print(
        f"Entities with all true matches covered : "
        f"{entities_with_all_matches:,}/{total_entities:,} "
        f"({entities_with_all_matches / total_entities * 100:.2f}%)"
    )

    return {
        "true_pairs": true_pairs,
        "retrieved_pairs": retrieved_pairs,
        "missed_pairs": missed_pairs,
        "candidate_count": candidate_count,
        "recall": pair_recall,
    }


# ============================================================
# FORENSIC FEATURE ANALYSIS
# ============================================================

def forensic_analysis(conn):

    cur = conn.cursor()

    print("\n" + "=" * 80)
    print("V10 FORENSIC ANALYSIS - MISSED TRUE PAIRS")
    print("=" * 80)

    # --------------------------------------------------------
    # Create missed pair table
    # --------------------------------------------------------

    cur.execute("""
        CREATE TEMP TABLE v10_missed AS
        SELECT
            g.source1_entity_id,
            g.matched_entity_id
        FROM v10_gt g
        LEFT JOIN v10_candidates c
          ON c.source1_entity_id = g.source1_entity_id
         AND c.candidate_entity_id = g.matched_entity_id
        WHERE c.candidate_entity_id IS NULL
    """)

    conn.commit()

    missed = cur.execute(
        "SELECT COUNT(*) FROM v10_missed"
    ).fetchone()[0]

    print(f"\nMissed true pairs: {missed:,}")

    if missed == 0:
        print("No missed pairs. V9 covers every sampled true pair.")
        return

    # --------------------------------------------------------
    # Feature overlap
    # --------------------------------------------------------

    feature_query = """
        SELECT

            SUM(
                CASE WHEN
                    s.name_key <> ''
                    AND r.name_key <> ''
                    AND s.name_key = r.name_key
                THEN 1 ELSE 0 END
            ) AS exact_name,

            SUM(
                CASE WHEN
                    s.sorted_name_key <> ''
                    AND r.sorted_name_key <> ''
                    AND s.sorted_name_key = r.sorted_name_key
                THEN 1 ELSE 0 END
            ) AS sorted_name,

            SUM(
                CASE WHEN
                    s.name_token_key <> ''
                    AND r.name_token_key <> ''
                    AND s.name_token_key = r.name_token_key
                THEN 1 ELSE 0 END
            ) AS token_name,

            SUM(
                CASE WHEN
                    s.first_name_token <> ''
                    AND r.first_name_token <> ''
                    AND s.first_name_token = r.first_name_token
                THEN 1 ELSE 0 END
            ) AS first_token,

            SUM(
                CASE WHEN
                    s.address_number <> ''
                    AND r.address_number <> ''
                    AND s.address_number = r.address_number
                THEN 1 ELSE 0 END
            ) AS address_number,

            SUM(
                CASE WHEN
                    s.address_key <> ''
                    AND r.address_key <> ''
                    AND s.address_key = r.address_key
                THEN 1 ELSE 0 END
            ) AS exact_address,

            SUM(
                CASE WHEN
                    s.address_token_sig <> ''
                    AND r.address_token_sig <> ''
                    AND s.address_token_sig = r.address_token_sig
                THEN 1 ELSE 0 END
            ) AS address_token,

            SUM(
                CASE WHEN
                    s.name_prefix_sig <> ''
                    AND r.name_prefix_sig <> ''
                    AND s.name_prefix_sig = r.name_prefix_sig
                THEN 1 ELSE 0 END
            ) AS name_prefix,

            SUM(
                CASE WHEN
                    s.name_suffix_sig <> ''
                    AND r.name_suffix_sig <> ''
                    AND s.name_suffix_sig = r.name_suffix_sig
                THEN 1 ELSE 0 END
            ) AS name_suffix

        FROM v10_missed m

        JOIN s1 s
          ON s.entity_id = m.source1_entity_id

        JOIN records r
          ON r.entity_id = m.matched_entity_id
    """

    row = cur.execute(feature_query).fetchone()

    feature_names = [
        ("Exact normalized name", row[0]),
        ("Sorted normalized name", row[1]),
        ("Exact name token set", row[2]),
        ("Same first name token", row[3]),
        ("Same address number", row[4]),
        ("Exact normalized address", row[5]),
        ("Same address token", row[6]),
        ("Same name prefix", row[7]),
        ("Same name suffix", row[8]),
    ]

    print("\nFeature overlap among MISSED true pairs:")

    for name, count in feature_names:
        pct = count / missed * 100
        print(
            f"{name:<30} "
            f"{count:>7,} "
            f"({pct:>6.2f}%)"
        )

    # --------------------------------------------------------
    # Combined signal categories
    # --------------------------------------------------------

    category_query = """
        SELECT
            CASE

                WHEN
                    (
                        s.name_key <> ''
                        AND r.name_key <> ''
                        AND s.name_key = r.name_key
                    )
                    OR
                    (
                        s.sorted_name_key <> ''
                        AND r.sorted_name_key <> ''
                        AND s.sorted_name_key = r.sorted_name_key
                    )
                    OR
                    (
                        s.name_token_key <> ''
                        AND r.name_token_key <> ''
                        AND s.name_token_key = r.name_token_key
                    )
                THEN 'strong_name_signal'

                WHEN
                    (
                        s.address_key <> ''
                        AND r.address_key <> ''
                        AND s.address_key = r.address_key
                    )
                    OR
                    (
                        s.address_token_sig <> ''
                        AND r.address_token_sig <> ''
                        AND s.address_token_sig = r.address_token_sig
                    )
                THEN 'strong_address_signal'

                WHEN
                    s.address_number <> ''
                    AND r.address_number <> ''
                    AND s.address_number = r.address_number
                    AND
                    (
                        s.name_prefix_sig = r.name_prefix_sig
                        OR
                        s.name_suffix_sig = r.name_suffix_sig
                    )
                THEN 'number_name_signature_signal'

                WHEN
                    s.first_name_token <> ''
                    AND r.first_name_token <> ''
                    AND s.first_name_token = r.first_name_token
                THEN 'first_token_signal'

                ELSE 'weak_or_no_exact_signal'

            END AS category,

            COUNT(*) AS cnt

        FROM v10_missed m

        JOIN s1 s
          ON s.entity_id = m.source1_entity_id

        JOIN records r
          ON r.entity_id = m.matched_entity_id

        GROUP BY category
        ORDER BY cnt DESC
    """

    print("\nMissed-pair signal categories:")

    categories = cur.execute(category_query).fetchall()

    for category, count in categories:

        pct = count / missed * 100

        print(
            f"{category:<35} "
            f"{count:>7,} "
            f"({pct:>6.2f}%)"
        )

    # --------------------------------------------------------
    # Strong address signal specifically
    # --------------------------------------------------------

    strong_address = cur.execute("""
        SELECT COUNT(*)
        FROM v10_missed m
        JOIN s1 s
          ON s.entity_id = m.source1_entity_id
        JOIN records r
          ON r.entity_id = m.matched_entity_id
        WHERE
            (
                s.address_key <> ''
                AND r.address_key <> ''
                AND s.address_key = r.address_key
            )
            OR
            (
                s.address_token_sig <> ''
                AND r.address_token_sig <> ''
                AND s.address_token_sig = r.address_token_sig
            )
            OR
            (
                s.address_number <> ''
                AND r.address_number <> ''
                AND s.address_number = r.address_number
            )
    """).fetchone()[0]

    print(
        f"\nMissed pairs with ANY address signal : "
        f"{strong_address:,} "
        f"({strong_address / missed * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Strong name signal specifically
    # --------------------------------------------------------

    strong_name = cur.execute("""
        SELECT COUNT(*)
        FROM v10_missed m
        JOIN s1 s
          ON s.entity_id = m.source1_entity_id
        JOIN records r
          ON r.entity_id = m.matched_entity_id
        WHERE
            (
                s.name_key <> ''
                AND r.name_key <> ''
                AND s.name_key = r.name_key
            )
            OR
            (
                s.sorted_name_key <> ''
                AND r.sorted_name_key <> ''
                AND s.sorted_name_key = r.sorted_name_key
            )
            OR
            (
                s.name_token_key <> ''
                AND r.name_token_key <> ''
                AND s.name_token_key = r.name_token_key
            )
            OR
            (
                s.first_name_token <> ''
                AND r.first_name_token <> ''
                AND s.first_name_token = r.first_name_token
            )
            OR
            (
                s.name_prefix_sig <> ''
                AND r.name_prefix_sig <> ''
                AND s.name_prefix_sig = r.name_prefix_sig
            )
            OR
            (
                s.name_suffix_sig <> ''
                AND r.name_suffix_sig <> ''
                AND s.name_suffix_sig = r.name_suffix_sig
            )
    """).fetchone()[0]

    print(
        f"Missed pairs with ANY name signal    : "
        f"{strong_name:,} "
        f"({strong_name / missed * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # No exact feature signal
    # --------------------------------------------------------

    no_signal = cur.execute("""
        SELECT COUNT(*)
        FROM v10_missed m
        JOIN s1 s
          ON s.entity_id = m.source1_entity_id
        JOIN records r
          ON r.entity_id = m.matched_entity_id
        WHERE NOT (

            (
                s.name_key <> ''
                AND r.name_key <> ''
                AND s.name_key = r.name_key
            )
            OR
            (
                s.sorted_name_key <> ''
                AND r.sorted_name_key <> ''
                AND s.sorted_name_key = r.sorted_name_key
            )
            OR
            (
                s.name_token_key <> ''
                AND r.name_token_key <> ''
                AND s.name_token_key = r.name_token_key
            )
            OR
            (
                s.first_name_token <> ''
                AND r.first_name_token <> ''
                AND s.first_name_token = r.first_name_token
            )
            OR
            (
                s.address_number <> ''
                AND r.address_number <> ''
                AND s.address_number = r.address_number
            )
            OR
            (
                s.address_key <> ''
                AND r.address_key <> ''
                AND s.address_key = r.address_key
            )
            OR
            (
                s.address_token_sig <> ''
                AND r.address_token_sig <> ''
                AND s.address_token_sig = r.address_token_sig
            )
            OR
            (
                s.name_prefix_sig <> ''
                AND r.name_prefix_sig <> ''
                AND s.name_prefix_sig = r.name_prefix_sig
            )
            OR
            (
                s.name_suffix_sig <> ''
                AND r.name_suffix_sig <> ''
                AND s.name_suffix_sig = r.name_suffix_sig
            )

        )
    """).fetchone()[0]

    print(
        f"No exact blocking-feature signal       : "
        f"{no_signal:,} "
        f"({no_signal / missed * 100:.2f}%)"
    )

    return missed


# ============================================================
# SHOW SAMPLE MISSES
# ============================================================

def show_sample_misses(conn, limit=20):

    cur = conn.cursor()

    print("\n" + "=" * 80)
    print(f"SAMPLE MISSED TRUE PAIRS ({limit})")
    print("=" * 80)

    # NOTE:
    # s1 contains normalized features, while records contains
    # original business_name/business_address.
    #
    # For S1 readable names/addresses, we try to load them from
    # train_source1.tsv if available.

    source1_path = (
        ROOT / "dataset" / "train" / "train_source1.tsv"
        if (ROOT / "dataset" / "train" / "train_source1.tsv").exists()
        else ROOT / "student_resource" / "dataset" / "train" / "train_source1.tsv"
    )

    if not source1_path.exists():
        print("\nRaw Source1 file not found:")
        print(source1_path)
        print("Skipping readable examples.")
        return

    cur.execute("""
        CREATE TEMP TABLE v10_raw_s1 (
            entity_id TEXT PRIMARY KEY,
            business_name TEXT,
            business_address TEXT,
            country TEXT
        )
    """)

    sample_ids = {
        row[0]
        for row in cur.execute(
            "SELECT source1_entity_id FROM v10_missed LIMIT ?",
            (limit,),
        )
    }

    if not sample_ids:
        print("No missed samples to display.")
        return

    with open(source1_path, "r", encoding="utf-8", newline="") as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            entity_id = row["entity_id"]

            if entity_id not in sample_ids:
                continue

            cur.execute(
                """
                INSERT OR IGNORE INTO v10_raw_s1
                (entity_id, business_name, business_address, country)
                VALUES (?, ?, ?, ?)
                """,
                (
                    entity_id,
                    row.get("business_name", ""),
                    row.get("business_address", ""),
                    row.get("country", ""),
                ),
            )

            if cur.execute(
                "SELECT COUNT(*) FROM v10_raw_s1"
            ).fetchone()[0] >= len(sample_ids):
                break

    conn.commit()
    print()

    rows = cur.execute("""
        SELECT
            m.source1_entity_id,
            m.matched_entity_id,

            s1.business_name,
            s1.business_address,

            r.business_name,
            r.business_address,

            CASE
                WHEN
                    s.address_token_sig <> ''
                    AND r.address_token_sig <> ''
                    AND s.address_token_sig = r.address_token_sig
                THEN 1
                ELSE 0
            END AS addr_token_same,

            CASE
                WHEN
                    s.address_number <> ''
                    AND r.address_number <> ''
                    AND s.address_number = r.address_number
                THEN 1
                ELSE 0
            END AS addr_number_same

        FROM v10_missed m

        JOIN s1 s
          ON s.entity_id = m.source1_entity_id

        JOIN records r
          ON r.entity_id = m.matched_entity_id

        LEFT JOIN v10_raw_s1 s1
          ON s1.entity_id = m.source1_entity_id

        LIMIT ?
    """, (limit,)).fetchall()

    for i, row in enumerate(rows, 1):

        (
            s1_id,
            s2_id,
            s1_name,
            s1_address,
            candidate_name,
            candidate_address,
            addr_token_same,
            addr_number_same,
        ) = row

        print(f"\n[{i}]")
        print(f"S1 ID       : {s1_id}")
        print(f"True ID     : {s2_id}")
        print(f"S1 Name     : {s1_name}")
        print(f"True Name   : {candidate_name}")
        print(f"S1 Address  : {s1_address}")
        print(f"True Address: {candidate_address}")
        print(f"Addr token  : {addr_token_same}")
        print(f"Addr number : {addr_number_same}")


# ============================================================
# MAIN
# ============================================================

def main():

    start_total = time.time()

    print("=" * 80)
    print("V10 FORENSIC BLOCKING ANALYSIS")
    print("=" * 80)

    print(f"\nDB : {DB_PATH}")
    print(f"GT : {GT_PATH}")
    print(f"Sample size : {SAMPLE_SIZE}")
    print(f"R18 frequency cap : {R18_CAP}")

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found:\n{DB_PATH}"
        )

    if not GT_PATH.exists():
        raise FileNotFoundError(
            f"Ground truth not found:\n{GT_PATH}"
        )

    conn = sqlite3.connect(
        str(DB_PATH),
        timeout=120,
    )

    # Faster temporary operations.
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -100000")

    try:

        load_ground_truth(conn, SAMPLE_SIZE)

        build_v9_candidates(conn)

        calculate_v9_recall(conn)

        forensic_analysis(conn)

        show_sample_misses(conn, limit=20)

        print("\n" + "=" * 80)
        print("V10 COMPLETE")
        print("=" * 80)

        print(
            f"Total runtime: "
            f"{(time.time() - start_total) / 60:.2f} minutes"
        )

        print(
            "\nIMPORTANT:"
            "\nV10 is forensic only."
            "\nIt does NOT create a new blocking version."
            "\nUse the missed-pair feature analysis to decide whether"
            "\nan additional V11 blocking rule is justified."
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()