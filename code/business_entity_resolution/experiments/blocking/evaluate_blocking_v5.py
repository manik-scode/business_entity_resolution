import csv
import sqlite3
import time
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "results" / "blocking" / "blocking_train_v2.db"
GT_PATH = ROOT / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"

SAMPLE_SIZE = 1000


# ============================================================
# HELPERS
# ============================================================

def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0

    k = (len(values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)

    if f == c:
        return values[f]

    return values[f] + (values[c] - values[f]) * (k - f)


def load_ground_truth(path, sample_ids):
    """
    Load only the ground-truth rows required for our 1000 S1 sample.
    """

    ground_truth = {}

    with open(path, "r", encoding="utf-8", newline="") as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            sid = row["source1_entity_id"]

            if sid not in sample_ids:
                continue

            raw = (row["matched_entity_ids"] or "").strip()

            if raw:
                matches = {
                    x.strip()
                    for x in raw.split(",")
                    if x.strip()
                }
            else:
                matches = set()

            ground_truth[sid] = matches

            if len(ground_truth) == len(sample_ids):
                break

    return ground_truth


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AMAZON ML CHALLENGE — V5 BLOCKING EVALUATION")
    print("=" * 70)

    print(f"\nDB      : {DB_PATH}")
    print(f"GT      : {GT_PATH}")
    print(f"Sample  : {SAMPLE_SIZE}")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    if not GT_PATH.exists():
        raise FileNotFoundError(f"Ground truth not found: {GT_PATH}")

    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    print("\n[1/5] Opening SQLite database...")

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-300000")

    # --------------------------------------------------------
    # CHECK DATABASE
    # --------------------------------------------------------

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

    if "records" not in tables:
        raise RuntimeError("records table not found.")

    if "s1" not in tables:
        raise RuntimeError("s1 table not found.")

    print("Database OK.")
    print("records table found.")
    print("s1 table found.")

    s1_count = conn.execute(
        "SELECT COUNT(*) FROM s1"
    ).fetchone()[0]

    print(f"Total S1 rows in DB: {s1_count:,}")

    # --------------------------------------------------------
    # CREATE TEMPORARY SAMPLE
    # --------------------------------------------------------

    print("\n[2/5] Selecting first 1000 S1 rows...")

    conn.execute("DROP TABLE IF EXISTS temp.v5_s1")
    conn.execute("DROP TABLE IF EXISTS temp.v5_candidates")

    conn.execute(f"""
        CREATE TEMP TABLE v5_s1 AS
        SELECT *
        FROM s1
        ORDER BY rowid
        LIMIT {SAMPLE_SIZE}
    """)

    conn.execute("""
        CREATE INDEX v5_s1_name
        ON v5_s1(country, name_key)
    """)

    conn.execute("""
        CREATE INDEX v5_s1_sorted
        ON v5_s1(country, sorted_name_key)
    """)

    conn.execute("""
        CREATE INDEX v5_s1_token_number
        ON v5_s1(country, name_token_key, address_number)
    """)

    conn.execute("""
        CREATE INDEX v5_s1_first_number
        ON v5_s1(country, first_name_token, address_number)
    """)

    conn.execute("""
        CREATE INDEX v5_s1_prefix_number
        ON v5_s1(country, name_prefix_sig, address_number)
    """)

    conn.execute("""
        CREATE INDEX v5_s1_suffix_number
        ON v5_s1(country, name_suffix_sig, address_number)
    """)

    conn.execute("""
        CREATE INDEX v5_s1_address
        ON v5_s1(country, address_key, first_name_token)
    """)

    conn.execute("""
        CREATE INDEX v5_s1_prefix_address
        ON v5_s1(country, name_prefix_sig, address_token_sig)
    """)

    conn.execute("""
        CREATE INDEX v5_s1_suffix_address
        ON v5_s1(country, name_suffix_sig, address_token_sig)
    """)

    conn.execute("""
        CREATE INDEX v5_s1_first_address
        ON v5_s1(country, first_name_token, address_token_sig)
    """)

    conn.execute("""
        CREATE TEMP TABLE v5_candidates (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,
            PRIMARY KEY (
                source1_entity_id,
                candidate_entity_id
            )
        )
    """)

    conn.execute("""
        CREATE INDEX v5_candidates_s1
        ON v5_candidates(source1_entity_id)
    """)

    sample_ids = {
        row[0]
        for row in conn.execute(
            "SELECT entity_id FROM v5_s1"
        )
    }

    print(f"Sample loaded: {len(sample_ids):,}")

    # --------------------------------------------------------
    # LOAD GROUND TRUTH
    # --------------------------------------------------------

    print("\n[3/5] Loading ground truth...")

    ground_truth = load_ground_truth(
        GT_PATH,
        sample_ids
    )

    print(
        f"Ground-truth rows loaded: "
        f"{len(ground_truth):,}"
    )

    missing_gt = sample_ids - set(ground_truth)

    if missing_gt:
        print(
            f"WARNING: {len(missing_gt)} "
            f"S1 rows missing from ground truth."
        )

    # --------------------------------------------------------
    # V5 RULES
    # --------------------------------------------------------

    rules = [

        (
            "R1_EXACT_NAME",
            """
            INSERT OR IGNORE INTO v5_candidates
            SELECT
                s.entity_id,
                r.entity_id
            FROM v5_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_key = s.name_key
            WHERE r.source IN ('S2', 'S3')
              AND s.name_key IS NOT NULL
              AND s.name_key <> ''
              AND r.name_key IS NOT NULL
              AND r.name_key <> ''
            """
        ),

        (
            "R2_SORTED_NAME",
            """
            INSERT OR IGNORE INTO v5_candidates
            SELECT
                s.entity_id,
                r.entity_id
            FROM v5_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.sorted_name_key = s.sorted_name_key
            WHERE r.source IN ('S2', 'S3')
              AND s.sorted_name_key IS NOT NULL
              AND s.sorted_name_key <> ''
              AND r.sorted_name_key IS NOT NULL
              AND r.sorted_name_key <> ''
            """
        ),

        (
            "R3_NAME_TOKEN_NUMBER",
            """
            INSERT OR IGNORE INTO v5_candidates
            SELECT
                s.entity_id,
                r.entity_id
            FROM v5_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_token_key = s.name_token_key
             AND r.address_number = s.address_number
            WHERE r.source IN ('S2', 'S3')
              AND s.name_token_key IS NOT NULL
              AND s.name_token_key <> ''
              AND s.address_number IS NOT NULL
              AND s.address_number <> ''
            """
        ),

        (
            "R4_FIRST_TOKEN_NUMBER",
            """
            INSERT OR IGNORE INTO v5_candidates
            SELECT
                s.entity_id,
                r.entity_id
            FROM v5_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.first_name_token = s.first_name_token
             AND r.address_number = s.address_number
            WHERE r.source IN ('S2', 'S3')
              AND s.first_name_token IS NOT NULL
              AND s.first_name_token <> ''
              AND s.address_number IS NOT NULL
              AND s.address_number <> ''
            """
        ),

        (
            "R5_FIRST_NUMBER_PREFIX",
            """
            INSERT OR IGNORE INTO v5_candidates
            SELECT
                s.entity_id,
                r.entity_id
            FROM v5_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.first_name_token = s.first_name_token
             AND r.address_number = s.address_number
             AND r.name_prefix_sig = s.name_prefix_sig
            WHERE r.source IN ('S2', 'S3')
              AND s.first_name_token IS NOT NULL
              AND s.first_name_token <> ''
              AND s.address_number IS NOT NULL
              AND s.address_number <> ''
              AND s.name_prefix_sig IS NOT NULL
              AND s.name_prefix_sig <> ''
            """
        ),

        (
            "R6_FIRST_NUMBER_SUFFIX",
            """
            INSERT OR IGNORE INTO v5_candidates
            SELECT
                s.entity_id,
                r.entity_id
            FROM v5_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.first_name_token = s.first_name_token
             AND r.address_number = s.address_number
             AND r.name_suffix_sig = s.name_suffix_sig
            WHERE r.source IN ('S2', 'S3')
              AND s.first_name_token IS NOT NULL
              AND s.first_name_token <> ''
              AND s.address_number IS NOT NULL
              AND s.address_number <> ''
              AND s.name_suffix_sig IS NOT NULL
              AND s.name_suffix_sig <> ''
            """
        ),

        (
            "R7_EXACT_ADDRESS_FIRST_TOKEN",
            """
            INSERT OR IGNORE INTO v5_candidates
            SELECT
                s.entity_id,
                r.entity_id
            FROM v5_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.address_key = s.address_key
             AND r.first_name_token = s.first_name_token
            WHERE r.source IN ('S2', 'S3')
              AND s.address_key IS NOT NULL
              AND s.address_key <> ''
              AND s.first_name_token IS NOT NULL
              AND s.first_name_token <> ''
            """
        ),

        (
            "R8_PREFIX_ADDRESS_TOKEN",
            """
            INSERT OR IGNORE INTO v5_candidates
            SELECT
                s.entity_id,
                r.entity_id
            FROM v5_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_prefix_sig = s.name_prefix_sig
             AND r.address_token_sig = s.address_token_sig
            WHERE r.source IN ('S2', 'S3')
              AND s.name_prefix_sig IS NOT NULL
              AND s.name_prefix_sig <> ''
              AND s.address_token_sig IS NOT NULL
              AND s.address_token_sig <> ''
            """
        ),

        (
            "R9_SUFFIX_ADDRESS_TOKEN",
            """
            INSERT OR IGNORE INTO v5_candidates
            SELECT
                s.entity_id,
                r.entity_id
            FROM v5_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_suffix_sig = s.name_suffix_sig
             AND r.address_token_sig = s.address_token_sig
            WHERE r.source IN ('S2', 'S3')
              AND s.name_suffix_sig IS NOT NULL
              AND s.name_suffix_sig <> ''
              AND s.address_token_sig IS NOT NULL
              AND s.address_token_sig <> ''
            """
        ),

        (
            "R10_FIRST_TOKEN_ADDRESS_TOKEN",
            """
            INSERT OR IGNORE INTO v5_candidates
            SELECT
                s.entity_id,
                r.entity_id
            FROM v5_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.first_name_token = s.first_name_token
             AND r.address_token_sig = s.address_token_sig
            WHERE r.source IN ('S2', 'S3')
              AND s.first_name_token IS NOT NULL
              AND s.first_name_token <> ''
              AND s.address_token_sig IS NOT NULL
              AND s.address_token_sig <> ''
            """
        ),

    ]

    # --------------------------------------------------------
    # RUN RULES
    # --------------------------------------------------------

    print("\n[4/5] Running V5 blocking rules...")
    print("-" * 70)

    for rule_name, sql in rules:

        before = conn.execute(
            "SELECT COUNT(*) FROM v5_candidates"
        ).fetchone()[0]

        start = time.time()

        conn.execute(sql)
        conn.commit()

        elapsed = time.time() - start

        after = conn.execute(
            "SELECT COUNT(*) FROM v5_candidates"
        ).fetchone()[0]

        added = after - before

        print(
            f"{rule_name:<35} "
            f"+{added:>10,}   "
            f"total={after:>12,}   "
            f"{elapsed:>7.2f}s"
        )

    # --------------------------------------------------------
    # BUILD CANDIDATE DICTIONARY
    # --------------------------------------------------------

    candidates = {}

    for sid, cid in conn.execute("""
        SELECT source1_entity_id, candidate_entity_id
        FROM v5_candidates
    """):

        candidates.setdefault(sid, set()).add(cid)

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    candidate_counts = [
        len(candidates.get(sid, set()))
        for sid in sample_ids
    ]

    total_candidates = sum(candidate_counts)

    nonempty_truth = [
        sid
        for sid in sample_ids
        if ground_truth.get(sid)
    ]

    true_pairs = sum(
        len(ground_truth.get(sid, set()))
        for sid in sample_ids
    )

    retrieved_true_pairs = 0

    covered_entities = 0

    for sid in nonempty_truth:

        truth = ground_truth.get(sid, set())
        cand = candidates.get(sid, set())

        retrieved_true_pairs += len(
            truth & cand
        )

        if truth.issubset(cand):
            covered_entities += 1

    pair_recall = (
        retrieved_true_pairs / true_pairs
        if true_pairs
        else 0
    )

    pair_precision = (
        retrieved_true_pairs / total_candidates
        if total_candidates
        else 0
    )

    no_candidate = sum(
        1 for x in candidate_counts
        if x == 0
    )

    no_match_rows = [
        sid
        for sid in sample_ids
        if not ground_truth.get(sid, set())
    ]

    no_match_with_candidates = sum(
        1
        for sid in no_match_rows
        if candidates.get(sid, set())
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("V5 BLOCKING RESULTS")
    print("=" * 70)

    print(
        f"\nSample S1 rows              : "
        f"{len(sample_ids):,}"
    )

    print(
        f"True matching pairs         : "
        f"{true_pairs:,}"
    )

    print(
        f"Retrieved true pairs        : "
        f"{retrieved_true_pairs:,}"
    )

    print(
        f"Candidate pairs             : "
        f"{total_candidates:,}"
    )

    print(
        f"\nCandidate Pair Recall       : "
        f"{pair_recall * 100:.4f}%"
    )

    print(
        f"Candidate Pair Precision    : "
        f"{pair_precision * 100:.4f}%"
    )

    print(
        f"All true matches covered    : "
        f"{covered_entities:,}/{len(nonempty_truth):,} "
        f"({(covered_entities / len(nonempty_truth) * 100) if nonempty_truth else 0:.2f}%)"
    )

    print(
        f"\nAverage candidates / S1     : "
        f"{sum(candidate_counts) / len(candidate_counts):.2f}"
    )

    print(
        f"P50 candidates              : "
        f"{percentile(candidate_counts, 50):.0f}"
    )

    print(
        f"P90 candidates              : "
        f"{percentile(candidate_counts, 90):.0f}"
    )

    print(
        f"P95 candidates              : "
        f"{percentile(candidate_counts, 95):.0f}"
    )

    print(
        f"P99 candidates              : "
        f"{percentile(candidate_counts, 99):.0f}"
    )

    print(
        f"Maximum candidates          : "
        f"{max(candidate_counts):,}"
    )

    print(
        f"Zero-candidate S1           : "
        f"{no_candidate:,}"
    )

    print(
        f"No-match S1 rows            : "
        f"{len(no_match_rows):,}"
    )

    print(
        f"No-match rows with candidate: "
        f"{no_match_with_candidates:,}"
    )

    print("\n" + "=" * 70)

    # --------------------------------------------------------
    # DECISION GUIDE
    # --------------------------------------------------------

    print("\nINTERPRETATION:")

    if pair_recall >= 0.90:
        print("[OK] Recall >= 90% -- blocking is strong enough to move toward ML.")

    elif pair_recall >= 0.80:
        print("[WARN] Recall 80-90% -- blocking is usable, but improvement may help.")

    else:
        print("[INFO] Recall < 80% -- improve blocking before training the ML matcher.")

    if total_candidates / len(sample_ids) <= 1000:
        print("[OK] Candidate volume is manageable.")

    elif total_candidates / len(sample_ids) <= 5000:
        print("[WARN] Candidate volume is somewhat high.")

    else:
        print("[WARN] Candidate volume is high -- avoid using this directly for full test.")

    print("\nTemporary V5 tables will disappear when the script exits.")
    print("Your original blocking_train_v2.db is NOT modified.")

    conn.close()


if __name__ == "__main__":
    main()
