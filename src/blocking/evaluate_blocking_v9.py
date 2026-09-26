import sqlite3
import sys
import time
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "results" / "blocking" / "blocking_train_v2.db"
GT_PATH = (
    ROOT / "dataset" / "train" / "ground_truth.tsv"
    if (ROOT / "dataset" / "train" / "ground_truth.tsv").exists()
    else ROOT / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"
)

SAMPLE_SIZE = 1000

# We will test how many records are allowed for an
# address_number + address_token_sig blocking key.
THRESHOLDS = [25, 50, 100, 250, 500, 1000, 2000, 5000]


def percentile(values, p):
    if not values:
        return 0

    values = sorted(values)
    k = (len(values) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(values) - 1)

    if f == c:
        return values[f]

    return values[f] + (values[c] - values[f]) * (k - f)


def load_ground_truth(path, sample_ids):
    gt = {}

    wanted = set(sample_ids)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")

        idx_s1 = header.index("source1_entity_id")
        idx_match = header.index("matched_entity_ids")

        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) <= max(idx_s1, idx_match):
                continue

            s1_id = parts[idx_s1]

            if s1_id not in wanted:
                continue

            raw = parts[idx_match].strip()

            if raw:
                gt[s1_id] = set(
                    x.strip()
                    for x in raw.split(",")
                    if x.strip()
                )
            else:
                gt[s1_id] = set()

            if len(gt) == len(wanted):
                break

    return gt


def evaluate_candidates(conn, candidate_table, gt):
    cur = conn.cursor()

    total_candidates = cur.execute(
        f"SELECT COUNT(*) FROM {candidate_table}"
    ).fetchone()[0]

    rows = cur.execute(
        f"""
        SELECT source1_entity_id, candidate_entity_id
        FROM {candidate_table}
        """
    )

    retrieved_true = 0
    candidates_per_s1 = defaultdict(int)

    for s1_id, candidate_id in rows:
        candidates_per_s1[s1_id] += 1

        if candidate_id in gt.get(s1_id, set()):
            retrieved_true += 1

    total_true = sum(len(v) for v in gt.values())

    recall = (
        retrieved_true / total_true * 100
        if total_true
        else 0
    )

    precision = (
        retrieved_true / total_candidates * 100
        if total_candidates
        else 0
    )

    counts = [
        candidates_per_s1.get(s1_id, 0)
        for s1_id in gt
    ]

    zero = sum(1 for x in counts if x == 0)

    return {
        "candidates": total_candidates,
        "retrieved_true": retrieved_true,
        "recall": recall,
        "precision": precision,
        "avg": (
            total_candidates / len(gt)
            if gt
            else 0
        ),
        "p50": percentile(counts, 50),
        "p90": percentile(counts, 90),
        "p95": percentile(counts, 95),
        "p99": percentile(counts, 99),
        "max": max(counts) if counts else 0,
        "zero": zero,
    }


def main():

    print("=" * 80)
    print("AMAZON ML CHALLENGE — V9 ADDRESS KEY FREQUENCY ANALYSIS")
    print("=" * 80)

    print()
    print("Opening:")
    print(DB_PATH)

    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    if not GT_PATH.exists():
        raise FileNotFoundError(GT_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-300000")

    cur = conn.cursor()

    # ------------------------------------------------------------
    # SAMPLE S1
    # ------------------------------------------------------------

    print()
    print(f"Loading first {SAMPLE_SIZE:,} S1 rows...")

    cur.execute("DROP TABLE IF EXISTS temp.v9_s1")

    cur.execute(
        f"""
        CREATE TEMP TABLE v9_s1 AS
        SELECT *
        FROM s1
        ORDER BY rowid
        LIMIT {SAMPLE_SIZE}
        """
    )

    cur.execute(
        """
        CREATE INDEX idx_v9_s1_id
        ON v9_s1(entity_id)
        """
    )

    cur.execute(
        """
        CREATE INDEX idx_v9_s1_addr_key
        ON v9_s1(country, address_number, address_token_sig)
        """
    )

    sample_ids = [
        row[0]
        for row in cur.execute(
            "SELECT entity_id FROM v9_s1"
        )
    ]

    print("Sample S1:", len(sample_ids))

    gt = load_ground_truth(GT_PATH, sample_ids)

    print("Ground truth rows loaded:", len(gt))

    total_true = sum(len(x) for x in gt.values())

    print("True positive pairs:", total_true)

    # ------------------------------------------------------------
    # BUILD DISTINCT R18 KEYS FROM SAMPLE
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("BUILDING ADDRESS NUMBER + ADDRESS TOKEN KEY FREQUENCIES")
    print("=" * 80)

    cur.execute("DROP TABLE IF EXISTS temp.v9_keys")

    cur.execute(
        """
        CREATE TEMP TABLE v9_keys AS
        SELECT DISTINCT
            country,
            address_number,
            address_token_sig
        FROM v9_s1
        WHERE country IS NOT NULL
          AND country <> ''
          AND address_number IS NOT NULL
          AND address_number <> ''
          AND address_token_sig IS NOT NULL
          AND address_token_sig <> ''
        """
    )

    cur.execute(
        """
        CREATE UNIQUE INDEX idx_v9_keys
        ON v9_keys(
            country,
            address_number,
            address_token_sig
        )
        """
    )

    key_count = cur.execute(
        "SELECT COUNT(*) FROM v9_keys"
    ).fetchone()[0]

    print("Unique sample keys:", key_count)

    # ------------------------------------------------------------
    # FREQUENCY OF EACH KEY IN S2 + S3
    # ------------------------------------------------------------

    print()
    print("Calculating key frequencies in S2/S3...")
    start = time.time()

    cur.execute("DROP TABLE IF EXISTS temp.v9_keyfreq")

    cur.execute(
        """
        CREATE TEMP TABLE v9_keyfreq AS
        SELECT
            k.country,
            k.address_number,
            k.address_token_sig,
            COUNT(*) AS frequency
        FROM v9_keys k
        CROSS JOIN records r
          ON r.country = k.country
         AND r.address_token_sig = k.address_token_sig
         AND r.address_number = k.address_number
        WHERE r.source IN ('S2', 'S3')
          AND r.address_number IS NOT NULL
          AND r.address_number <> ''
          AND r.address_token_sig IS NOT NULL
          AND r.address_token_sig <> ''
        GROUP BY
            k.country,
            k.address_number,
            k.address_token_sig
        """
    )

    cur.execute(
        """
        CREATE INDEX idx_v9_kf
        ON v9_keyfreq(
            country,
            address_number,
            address_token_sig,
            frequency
        )
        """
    )

    conn.commit()

    elapsed = time.time() - start

    print(f"Frequency calculation complete: {elapsed:.2f}s")

    # ------------------------------------------------------------
    # KEY FREQUENCY DISTRIBUTION
    # ------------------------------------------------------------

    frequencies = [
        row[0]
        for row in cur.execute(
            "SELECT frequency FROM v9_keyfreq"
        )
    ]

    print()
    print("Key frequency distribution:")
    print(f"  Keys found : {len(frequencies):,}")
    print(f"  P50        : {percentile(frequencies, 50):,.0f}")
    print(f"  P90        : {percentile(frequencies, 90):,.0f}")
    print(f"  P95        : {percentile(frequencies, 95):,.0f}")
    print(f"  P99        : {percentile(frequencies, 99):,.0f}")
    print(f"  Maximum    : {max(frequencies) if frequencies else 0:,}")

    # ------------------------------------------------------------
    # BUILD V5 BASELINE
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("BUILDING V5 BASELINE")
    print("=" * 80)

    cur.execute("DROP TABLE IF EXISTS temp.v9_base")

    cur.execute(
        """
        CREATE TEMP TABLE v9_base (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,
            PRIMARY KEY (
                source1_entity_id,
                candidate_entity_id
            )
        )
        """
    )

    rules = [

        # R1 exact name
        """
        INSERT OR IGNORE INTO v9_base
        SELECT s.entity_id, r.entity_id
        FROM v9_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.name_key = s.name_key
        WHERE r.source IN ('S2','S3')
          AND s.name_key IS NOT NULL
          AND s.name_key <> ''
          AND r.name_key IS NOT NULL
          AND r.name_key <> ''
        """,

        # R2 sorted name
        """
        INSERT OR IGNORE INTO v9_base
        SELECT s.entity_id, r.entity_id
        FROM v9_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.sorted_name_key = s.sorted_name_key
        WHERE r.source IN ('S2','S3')
          AND s.sorted_name_key IS NOT NULL
          AND s.sorted_name_key <> ''
          AND r.sorted_name_key IS NOT NULL
          AND r.sorted_name_key <> ''
        """,

        # R3 name token + number
        """
        INSERT OR IGNORE INTO v9_base
        SELECT s.entity_id, r.entity_id
        FROM v9_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.name_token_key = s.name_token_key
         AND r.address_number = s.address_number
        WHERE r.source IN ('S2','S3')
          AND s.name_token_key IS NOT NULL
          AND s.name_token_key <> ''
          AND s.address_number IS NOT NULL
          AND s.address_number <> ''
          AND r.name_token_key IS NOT NULL
          AND r.name_token_key <> ''
          AND r.address_number IS NOT NULL
          AND r.address_number <> ''
        """,

        # R4 first token + number
        """
        INSERT OR IGNORE INTO v9_base
        SELECT s.entity_id, r.entity_id
        FROM v9_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.first_name_token = s.first_name_token
         AND r.address_number = s.address_number
        WHERE r.source IN ('S2','S3')
          AND s.first_name_token IS NOT NULL
          AND s.first_name_token <> ''
          AND s.address_number IS NOT NULL
          AND s.address_number <> ''
          AND r.first_name_token IS NOT NULL
          AND r.first_name_token <> ''
          AND r.address_number IS NOT NULL
          AND r.address_number <> ''
        """,

        # R5 first + number + prefix
        """
        INSERT OR IGNORE INTO v9_base
        SELECT s.entity_id, r.entity_id
        FROM v9_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.first_name_token = s.first_name_token
         AND r.address_number = s.address_number
         AND r.name_prefix_sig = s.name_prefix_sig
        WHERE r.source IN ('S2','S3')
          AND s.first_name_token <> ''
          AND s.address_number <> ''
          AND s.name_prefix_sig <> ''
          AND r.first_name_token <> ''
          AND r.address_number <> ''
          AND r.name_prefix_sig <> ''
        """,

        # R6 first + number + suffix
        """
        INSERT OR IGNORE INTO v9_base
        SELECT s.entity_id, r.entity_id
        FROM v9_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.first_name_token = s.first_name_token
         AND r.address_number = s.address_number
         AND r.name_suffix_sig = s.name_suffix_sig
        WHERE r.source IN ('S2','S3')
          AND s.first_name_token <> ''
          AND s.address_number <> ''
          AND s.name_suffix_sig <> ''
          AND r.first_name_token <> ''
          AND r.address_number <> ''
          AND r.name_suffix_sig <> ''
        """,

        # R7 exact address + first token
        """
        INSERT OR IGNORE INTO v9_base
        SELECT s.entity_id, r.entity_id
        FROM v9_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.address_key = s.address_key
         AND r.first_name_token = s.first_name_token
        WHERE r.source IN ('S2','S3')
          AND s.address_key <> ''
          AND s.first_name_token <> ''
          AND r.address_key <> ''
          AND r.first_name_token <> ''
        """,

        # R8 prefix + address token
        """
        INSERT OR IGNORE INTO v9_base
        SELECT s.entity_id, r.entity_id
        FROM v9_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.name_prefix_sig = s.name_prefix_sig
         AND r.address_token_sig = s.address_token_sig
        WHERE r.source IN ('S2','S3')
          AND s.name_prefix_sig <> ''
          AND s.address_token_sig <> ''
          AND r.name_prefix_sig <> ''
          AND r.address_token_sig <> ''
        """,

        # R9 suffix + address token
        """
        INSERT OR IGNORE INTO v9_base
        SELECT s.entity_id, r.entity_id
        FROM v9_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.name_suffix_sig = s.name_suffix_sig
         AND r.address_token_sig = s.address_token_sig
        WHERE r.source IN ('S2','S3')
          AND s.name_suffix_sig <> ''
          AND s.address_token_sig <> ''
          AND r.name_suffix_sig <> ''
          AND r.address_token_sig <> ''
        """,

        # R10 first token + address token
        """
        INSERT OR IGNORE INTO v9_base
        SELECT s.entity_id, r.entity_id
        FROM v9_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.first_name_token = s.first_name_token
         AND r.address_token_sig = s.address_token_sig
        WHERE r.source IN ('S2','S3')
          AND s.first_name_token <> ''
          AND s.address_token_sig <> ''
          AND r.first_name_token <> ''
          AND r.address_token_sig <> ''
        """
    ]

    for i, sql in enumerate(rules, 1):
        start = time.time()
        cur.execute(sql)
        conn.commit()

        count = cur.execute(
            "SELECT COUNT(*) FROM v9_base"
        ).fetchone()[0]

        print(
            f"R{i:<2} complete "
            f"total={count:>10,} "
            f"time={time.time()-start:>7.2f}s"
        )

    base = evaluate_candidates(conn, "v9_base", gt)

    print()
    print("V5 BASELINE")
    print("-" * 60)
    print(f"Candidates : {base['candidates']:,}")
    print(f"Recall     : {base['recall']:.4f}%")
    print(f"Precision  : {base['precision']:.4f}%")
    print(f"Average    : {base['avg']:.2f}")
    print(f"P50/P90    : {base['p50']:.0f} / {base['p90']:.0f}")
    print(f"P95/P99    : {base['p95']:.0f} / {base['p99']:.0f}")
    print(f"Maximum    : {base['max']:,}")

    # ------------------------------------------------------------
    # TEST FREQUENCY CAPS
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("R18 FREQUENCY-CAPPED EXPERIMENT")
    print("=" * 80)

    print()
    print(
        f"{'CAP':>8} "
        f"{'ADDED':>12} "
        f"{'TOTAL':>12} "
        f"{'RECALL':>10} "
        f"{'DELTA':>10} "
        f"{'PRECISION':>11} "
        f"{'AVG':>10} "
        f"{'P99':>10}"
    )

    print("-" * 95)

    for cap in THRESHOLDS:

        cur.execute("DROP TABLE IF EXISTS temp.v9_candidates")

        cur.execute(
            """
            CREATE TEMP TABLE v9_candidates (
                source1_entity_id TEXT NOT NULL,
                candidate_entity_id TEXT NOT NULL,
                PRIMARY KEY (
                    source1_entity_id,
                    candidate_entity_id
                )
            )
            """
        )

        # Copy baseline
        cur.execute(
            """
            INSERT INTO v9_candidates
            SELECT *
            FROM v9_base
            """
        )

        start = time.time()

        # Frequency-capped R18
        cur.execute(
            """
            INSERT OR IGNORE INTO v9_candidates
            SELECT
                s.entity_id,
                r.entity_id
            FROM v9_s1 s
            JOIN v9_keyfreq k
              ON k.country = s.country
             AND k.address_number = s.address_number
             AND k.address_token_sig = s.address_token_sig
            JOIN records r
              ON r.country = k.country
             AND r.address_number = k.address_number
             AND r.address_token_sig = k.address_token_sig
            WHERE k.frequency <= ?
              AND r.source IN ('S2','S3')
              AND s.address_number <> ''
              AND s.address_token_sig <> ''
            """,
            (cap,)
        )

        conn.commit()

        result = evaluate_candidates(
            conn,
            "v9_candidates",
            gt
        )

        added = result["candidates"] - base["candidates"]
        delta = result["recall"] - base["recall"]

        print(
            f"{cap:>8,} "
            f"{added:>12,} "
            f"{result['candidates']:>12,} "
            f"{result['recall']:>9.4f}% "
            f"{delta:>+9.4f}% "
            f"{result['precision']:>10.4f}% "
            f"{result['avg']:>10.2f} "
            f"{result['p99']:>10.0f}"
        )

    # ------------------------------------------------------------
    # V9 OPTIMIZED UNION
    # V5 + R17 + R18(cap=500) + R19 + R20
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("V9 OPTIMIZED UNION")
    print("=" * 80)

    V9_CAP = 500

    cur.execute("DROP TABLE IF EXISTS temp.v9_final")

    cur.execute(
        """
        CREATE TEMP TABLE v9_final (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,
            PRIMARY KEY (
                source1_entity_id,
                candidate_entity_id
            )
        )
        """
    )

    # ------------------------------------------------------------
    # Start with V5 baseline
    # ------------------------------------------------------------

    cur.execute(
        """
        INSERT INTO v9_final
        SELECT *
        FROM v9_base
        """
    )

    conn.commit()

    base_count = cur.execute(
        "SELECT COUNT(*) FROM v9_final"
    ).fetchone()[0]

    print(
        f"V5 BASELINE"
        f" -> {base_count:,} candidates"
    )

    # ------------------------------------------------------------
    # R17 — EXACT ADDRESS
    # ------------------------------------------------------------

    before = base_count
    start = time.time()

    cur.execute(
        """
        INSERT OR IGNORE INTO v9_final
        SELECT
            s.entity_id,
            r.entity_id
        FROM v9_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.address_key = s.address_key
        WHERE r.source IN ('S2','S3')
          AND s.address_key IS NOT NULL
          AND s.address_key <> ''
          AND r.address_key IS NOT NULL
          AND r.address_key <> ''
        """
    )

    conn.commit()

    after = cur.execute(
        "SELECT COUNT(*) FROM v9_final"
    ).fetchone()[0]

    print(
        f"R17_EXACT_ADDRESS"
        f" -> +{after-before:,}"
        f" total={after:,}"
        f" time={time.time()-start:.2f}s"
    )

    # ------------------------------------------------------------
    # R18 — ADDRESS NUMBER + ADDRESS TOKEN
    # FREQUENCY CAP = 500
    # ------------------------------------------------------------

    before = after
    start = time.time()

    cur.execute(
        """
        INSERT OR IGNORE INTO v9_final
        SELECT
            s.entity_id,
            r.entity_id
        FROM v9_s1 s
        JOIN v9_keyfreq k
          ON k.country = s.country
         AND k.address_number = s.address_number
         AND k.address_token_sig = s.address_token_sig
        JOIN records r
          ON r.country = k.country
         AND r.address_number = k.address_number
         AND r.address_token_sig = k.address_token_sig
        WHERE k.frequency <= ?
          AND r.source IN ('S2','S3')
          AND s.address_number IS NOT NULL
          AND s.address_number <> ''
          AND s.address_token_sig IS NOT NULL
          AND s.address_token_sig <> ''
        """,
        (V9_CAP,)
    )

    conn.commit()

    after = cur.execute(
        "SELECT COUNT(*) FROM v9_final"
    ).fetchone()[0]

    print(
        f"R18_ADDR_NUMBER_TOKEN_CAP_{V9_CAP}"
        f" -> +{after-before:,}"
        f" total={after:,}"
        f" time={time.time()-start:.2f}s"
    )

    # ------------------------------------------------------------
    # R19 — PREFIX + NUMBER
    # ------------------------------------------------------------

    before = after
    start = time.time()

    cur.execute(
        """
        INSERT OR IGNORE INTO v9_final
        SELECT
            s.entity_id,
            r.entity_id
        FROM v9_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.name_prefix_sig = s.name_prefix_sig
         AND r.address_number = s.address_number
        WHERE r.source IN ('S2','S3')
          AND s.name_prefix_sig IS NOT NULL
          AND s.name_prefix_sig <> ''
          AND s.address_number IS NOT NULL
          AND s.address_number <> ''
          AND r.name_prefix_sig IS NOT NULL
          AND r.name_prefix_sig <> ''
          AND r.address_number IS NOT NULL
          AND r.address_number <> ''
        """
    )

    conn.commit()

    after = cur.execute(
        "SELECT COUNT(*) FROM v9_final"
    ).fetchone()[0]

    print(
        f"R19_PREFIX_NUMBER"
        f" -> +{after-before:,}"
        f" total={after:,}"
        f" time={time.time()-start:.2f}s"
    )

    # ------------------------------------------------------------
    # R20 — SUFFIX + NUMBER
    # ------------------------------------------------------------

    before = after
    start = time.time()

    cur.execute(
        """
        INSERT OR IGNORE INTO v9_final
        SELECT
            s.entity_id,
            r.entity_id
        FROM v9_s1 s
        JOIN records r
          ON r.country = s.country
         AND r.name_suffix_sig = s.name_suffix_sig
         AND r.address_number = s.address_number
        WHERE r.source IN ('S2','S3')
          AND s.name_suffix_sig IS NOT NULL
          AND s.name_suffix_sig <> ''
          AND s.address_number IS NOT NULL
          AND s.address_number <> ''
          AND r.name_suffix_sig IS NOT NULL
          AND r.name_suffix_sig <> ''
          AND r.address_number IS NOT NULL
          AND r.address_number <> ''
        """
    )

    conn.commit()

    after = cur.execute(
        "SELECT COUNT(*) FROM v9_final"
    ).fetchone()[0]

    print(
        f"R20_SUFFIX_NUMBER"
        f" -> +{after-before:,}"
        f" total={after:,}"
        f" time={time.time()-start:.2f}s"
    )

    # ------------------------------------------------------------
    # FINAL METRICS
    # ------------------------------------------------------------

    final = evaluate_candidates(
        conn,
        "v9_final",
        gt
    )

    print()
    print("=" * 80)
    print("V9 OPTIMIZED FINAL RESULTS")
    print("=" * 80)

    print(
        f"Candidate Pair Recall       : "
        f"{final['recall']:.4f}%"
    )

    print(
        f"Candidate Pair Precision    : "
        f"{final['precision']:.4f}%"
    )

    print(
        f"Total candidate pairs       : "
        f"{final['candidates']:,}"
    )

    print(
        f"Average candidates / S1     : "
        f"{final['avg']:.2f}"
    )

    print(
        f"Median (P50) candidates     : "
        f"{final['p50']:.0f}"
    )

    print(
        f"P90 candidates              : "
        f"{final['p90']:.0f}"
    )

    print(
        f"P95 candidates              : "
        f"{final['p95']:.0f}"
    )

    print(
        f"P99 candidates              : "
        f"{final['p99']:.0f}"
    )

    print(
        f"Maximum candidates          : "
        f"{final['max']:,}"
    )

    print(
        f"Zero-candidate S1           : "
        f"{final['zero']:,}"
    )

    print()
    print(
        "Comparison with V5:"
    )

    print(
        f"  Recall: "
        f"{base['recall']:.4f}% -> "
        f"{final['recall']:.4f}%"
    )

    print(
        f"  Candidates: "
        f"{base['candidates']:,} -> "
        f"{final['candidates']:,}"
    )

    print(
        f"  Avg/S1: "
        f"{base['avg']:.2f} -> "
        f"{final['avg']:.2f}"
    )

    print()
    print(
        "Comparison with V8 uncapped:"
    )

    print(
        "  V8 recall      : 84.3048%"
    )

    print(
        f"  V9 recall      : {final['recall']:.4f}%"
    )

    print(
        "  V8 candidates  : 405,698"
    )

    print(
        f"  V9 candidates  : {final['candidates']:,}"
    )

    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print("""
We are NOT creating a submission here.

We are trying to find the smallest useful frequency cap for:

    country + address_number + address_token_sig

A good cap should:

1. Keep most of R18's recall gain.
2. Remove a large number of false candidate pairs.
3. Keep P99 candidate volume manageable.

After this result we will decide whether R18 should be:
    - kept without a cap,
    - kept with a frequency cap,
    - or removed.

Then we either LOCK BLOCKING or perform one final targeted
blocking experiment before moving to the ML pair matcher.
""")

    conn.close()


if __name__ == "__main__":
    main()
