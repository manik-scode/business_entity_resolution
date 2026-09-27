import csv
import sqlite3
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "results" / "blocking" / "blocking_train_v2.db"
GT_PATH = ROOT / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"

SAMPLE_SIZE = 1000


def pct(values, p):
    values = sorted(values)
    if not values:
        return 0

    k = (len(values) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)

    if lo == hi:
        return values[lo]

    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def load_ground_truth(sample_ids):
    gt = {}

    with open(GT_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            sid = row["source1_entity_id"]
            if sid not in sample_ids:
                continue

            raw = (row["matched_entity_ids"] or "").strip()
            gt[sid] = (
                {x.strip() for x in raw.split(",") if x.strip()}
                if raw
                else set()
            )

            if len(gt) == len(sample_ids):
                break

    return gt


def calculate_metrics(conn, sample_ids, gt, table_name="v8_candidates"):
    candidates = {}

    for sid, cid in conn.execute(f"""
        SELECT source1_entity_id, candidate_entity_id
        FROM {table_name}
    """):
        candidates.setdefault(sid, set()).add(cid)

    counts = [
        len(candidates.get(sid, set()))
        for sid in sample_ids
    ]

    total_candidates = sum(counts)

    true_pairs = sum(
        len(gt.get(sid, set()))
        for sid in sample_ids
    )

    retrieved_true = 0

    nonempty_truth = [
        sid for sid in sample_ids
        if gt.get(sid, set())
    ]

    covered = 0

    for sid in nonempty_truth:
        truth = gt[sid]
        cand = candidates.get(sid, set())

        retrieved_true += len(truth & cand)

        if truth.issubset(cand):
            covered += 1

    recall = (
        retrieved_true / true_pairs
        if true_pairs else 0
    )

    precision = (
        retrieved_true / total_candidates
        if total_candidates else 0
    )

    return {
        "total_candidates": total_candidates,
        "true_pairs": true_pairs,
        "retrieved_true": retrieved_true,
        "recall": recall,
        "precision": precision,
        "covered": covered,
        "covered_total": len(nonempty_truth),
        "avg": total_candidates / len(sample_ids),
        "p50": pct(counts, 50),
        "p90": pct(counts, 90),
        "p95": pct(counts, 95),
        "p99": pct(counts, 99),
        "max": max(counts) if counts else 0,
        "zero": sum(x == 0 for x in counts),
    }


def main():
    print("=" * 80)
    print("AMAZON ML CHALLENGE — V8 BLOCKING EVALUATION (INCREMENTAL NEW RULES)")
    print("=" * 80)

    print("\nOpening existing V2 database...")
    print(DB_PATH)

    conn = sqlite3.connect(str(DB_PATH))

    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-300000")

    # ---------------------------------------------------------
    # SAMPLE
    # ---------------------------------------------------------
    conn.execute("DROP TABLE IF EXISTS temp.v8_s1")
    conn.execute("DROP TABLE IF EXISTS temp.v8_candidates")

    conn.execute(f"""
        CREATE TEMP TABLE v8_s1 AS
        SELECT *
        FROM s1
        ORDER BY rowid
        LIMIT {SAMPLE_SIZE}
    """)

    conn.execute("CREATE INDEX v8_s1_name ON v8_s1(country, name_key)")
    conn.execute("CREATE INDEX v8_s1_sorted ON v8_s1(country, sorted_name_key)")
    conn.execute("CREATE INDEX v8_s1_token_number ON v8_s1(country, name_token_key, address_number)")
    conn.execute("CREATE INDEX v8_s1_first_number ON v8_s1(country, first_name_token, address_number)")
    conn.execute("CREATE INDEX v8_s1_prefix_number ON v8_s1(country, name_prefix_sig, address_number)")
    conn.execute("CREATE INDEX v8_s1_suffix_number ON v8_s1(country, name_suffix_sig, address_number)")
    conn.execute("CREATE INDEX v8_s1_address ON v8_s1(country, address_key)")
    conn.execute("CREATE INDEX v8_s1_prefix_address ON v8_s1(country, name_prefix_sig, address_token_sig)")
    conn.execute("CREATE INDEX v8_s1_suffix_address ON v8_s1(country, name_suffix_sig, address_token_sig)")
    conn.execute("CREATE INDEX v8_s1_first_address ON v8_s1(country, first_name_token, address_token_sig)")
    conn.execute("CREATE INDEX v8_s1_token_address ON v8_s1(country, address_token_sig, address_number)")

    conn.execute("""
        CREATE TEMP TABLE v8_candidates (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,
            PRIMARY KEY (
                source1_entity_id,
                candidate_entity_id
            )
        )
    """)

    conn.execute("""
        CREATE INDEX v8_candidates_s1
        ON v8_candidates(source1_entity_id)
    """)

    sample_ids = {
        row[0]
        for row in conn.execute(
            "SELECT entity_id FROM v8_s1"
        )
    }

    print(f"\nSample S1: {len(sample_ids):,}")

    # ---------------------------------------------------------
    # GROUND TRUTH
    # ---------------------------------------------------------
    gt = load_ground_truth(sample_ids)
    print(f"Ground truth rows loaded: {len(gt):,}")

    # ---------------------------------------------------------
    # BASELINE V5 RULES
    # ---------------------------------------------------------
    base_rules = [
        (
            "R1_EXACT_NAME",
            """
            INSERT OR IGNORE INTO v8_candidates
            SELECT s.entity_id, r.entity_id
            FROM v8_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_key = s.name_key
            WHERE r.source IN ('S2','S3')
              AND s.name_key <> ''
            """
        ),
        (
            "R2_SORTED_NAME",
            """
            INSERT OR IGNORE INTO v8_candidates
            SELECT s.entity_id, r.entity_id
            FROM v8_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.sorted_name_key = s.sorted_name_key
            WHERE r.source IN ('S2','S3')
              AND s.sorted_name_key <> ''
            """
        ),
        (
            "R3_NAME_TOKEN_NUMBER",
            """
            INSERT OR IGNORE INTO v8_candidates
            SELECT s.entity_id, r.entity_id
            FROM v8_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_token_key = s.name_token_key
             AND r.address_number = s.address_number
            WHERE r.source IN ('S2','S3')
              AND s.name_token_key <> ''
              AND s.address_number <> ''
            """
        ),
        (
            "R4_FIRST_TOKEN_NUMBER",
            """
            INSERT OR IGNORE INTO v8_candidates
            SELECT s.entity_id, r.entity_id
            FROM v8_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.first_name_token = s.first_name_token
             AND r.address_number = s.address_number
            WHERE r.source IN ('S2','S3')
              AND s.first_name_token <> ''
              AND s.address_number <> ''
            """
        ),
        (
            "R5_FIRST_NUMBER_PREFIX",
            """
            INSERT OR IGNORE INTO v8_candidates
            SELECT s.entity_id, r.entity_id
            FROM v8_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.first_name_token = s.first_name_token
             AND r.address_number = s.address_number
             AND r.name_prefix_sig = s.name_prefix_sig
            WHERE r.source IN ('S2','S3')
              AND s.first_name_token <> ''
              AND s.address_number <> ''
              AND s.name_prefix_sig <> ''
            """
        ),
        (
            "R6_FIRST_NUMBER_SUFFIX",
            """
            INSERT OR IGNORE INTO v8_candidates
            SELECT s.entity_id, r.entity_id
            FROM v8_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.first_name_token = s.first_name_token
             AND r.address_number = s.address_number
             AND r.name_suffix_sig = s.name_suffix_sig
            WHERE r.source IN ('S2','S3')
              AND s.first_name_token <> ''
              AND s.address_number <> ''
              AND s.name_suffix_sig <> ''
            """
        ),
        (
            "R7_EXACT_ADDRESS_FIRST",
            """
            INSERT OR IGNORE INTO v8_candidates
            SELECT s.entity_id, r.entity_id
            FROM v8_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.address_key = s.address_key
             AND r.first_name_token = s.first_name_token
            WHERE r.source IN ('S2','S3')
              AND s.address_key <> ''
              AND s.first_name_token <> ''
            """
        ),
        (
            "R8_PREFIX_ADDRESS",
            """
            INSERT OR IGNORE INTO v8_candidates
            SELECT s.entity_id, r.entity_id
            FROM v8_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_prefix_sig = s.name_prefix_sig
             AND r.address_token_sig = s.address_token_sig
            WHERE r.source IN ('S2','S3')
              AND s.name_prefix_sig <> ''
              AND s.address_token_sig <> ''
            """
        ),
        (
            "R9_SUFFIX_ADDRESS",
            """
            INSERT OR IGNORE INTO v8_candidates
            SELECT s.entity_id, r.entity_id
            FROM v8_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_suffix_sig = s.name_suffix_sig
             AND r.address_token_sig = s.address_token_sig
            WHERE r.source IN ('S2','S3')
              AND s.name_suffix_sig <> ''
              AND s.address_token_sig <> ''
            """
        ),
        (
            "R10_FIRST_ADDRESS",
            """
            INSERT OR IGNORE INTO v8_candidates
            SELECT s.entity_id, r.entity_id
            FROM v8_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.first_name_token = s.first_name_token
             AND r.address_token_sig = s.address_token_sig
            WHERE r.source IN ('S2','S3')
              AND s.first_name_token <> ''
              AND s.address_token_sig <> ''
            """
        ),
    ]

    print("\n" + "=" * 80)
    print("RUNNING V5 BASELINE (R1 to R10)")
    print("=" * 80)

    for name, sql in base_rules:
        start = time.time()
        before = conn.execute("SELECT COUNT(*) FROM v8_candidates").fetchone()[0]
        conn.execute(sql)
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM v8_candidates").fetchone()[0]
        print(f"{name:<35} +{after-before:>10,} total={after:>12,} time={time.time()-start:>7.2f}s")

    base_metrics = calculate_metrics(conn, sample_ids, gt, "v8_candidates")

    print("\n" + "-" * 80)
    print("V5 BASELINE RESULTS:")
    print(f"  Recall     : {base_metrics['recall']*100:.4f}%")
    print(f"  Precision  : {base_metrics['precision']*100:.4f}%")
    print(f"  Candidates : {base_metrics['total_candidates']:,}")
    print(f"  Average/S1 : {base_metrics['avg']:.2f}")
    print(f"  P50/P90/P95: {base_metrics['p50']:.0f} / {base_metrics['p90']:.0f} / {base_metrics['p95']:.0f}")
    print(f"  Max        : {base_metrics['max']:,}")
    print(f"  Zero S1    : {base_metrics['zero']}")
    print("-" * 80)

    # ---------------------------------------------------------
    # NEW RULES TO TEST
    # ---------------------------------------------------------
    new_candidate_rules = [
        (
            "R17_EXACT_ADDRESS",
            """
            INSERT OR IGNORE INTO v8_candidates
            SELECT s.entity_id, r.entity_id
            FROM v8_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.address_key = s.address_key
            WHERE r.source IN ('S2','S3')
              AND s.address_key <> ''
            """
        ),
        (
            "R18_ADDRESS_NUMBER_TOKEN",
            """
            INSERT OR IGNORE INTO v8_candidates
            SELECT s.entity_id, r.entity_id
            FROM v8_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.address_token_sig = s.address_token_sig
             AND r.address_number = s.address_number
            WHERE r.source IN ('S2','S3')
              AND s.address_number <> ''
              AND s.address_token_sig <> ''
            """
        ),
        (
            "R19_PREFIX_NUMBER",
            """
            INSERT OR IGNORE INTO v8_candidates
            SELECT s.entity_id, r.entity_id
            FROM v8_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_prefix_sig = s.name_prefix_sig
             AND r.address_number = s.address_number
            WHERE r.source IN ('S2','S3')
              AND s.name_prefix_sig <> ''
              AND s.address_number <> ''
            """
        ),
        (
            "R20_SUFFIX_NUMBER",
            """
            INSERT OR IGNORE INTO v8_candidates
            SELECT s.entity_id, r.entity_id
            FROM v8_s1 s
            JOIN records r
              ON r.country = s.country
             AND r.name_suffix_sig = s.name_suffix_sig
             AND r.address_number = s.address_number
            WHERE r.source IN ('S2','S3')
              AND s.name_suffix_sig <> ''
              AND s.address_number <> ''
            """
        ),
    ]

    # ---------------------------------------------------------
    # PART A: STANDALONE EVALUATION OF EACH RULE ON TOP OF V5
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("STANDALONE TEST: V5 BASELINE + EACH NEW RULE INDIVIDUALLY")
    print("=" * 80)
    print(f"{'STANDALONE RULE':<32} {'ADDED CAND':>12} {'CANDIDATES':>12} {'RECALL':>10} {'DELTA REC':>11} {'AVG/S1':>10} {'TIME':>8}")
    print("-" * 100)

    for name, sql in new_candidate_rules:
        # Create temp candidate table copying baseline
        conn.execute("DROP TABLE IF EXISTS temp.v8_cand_test")
        conn.execute("CREATE TEMP TABLE v8_cand_test AS SELECT * FROM v8_candidates")

        t0 = time.time()
        before_cnt = conn.execute("SELECT COUNT(*) FROM v8_cand_test").fetchone()[0]

        # Apply single rule
        rule_sql = sql.replace("v8_candidates", "v8_cand_test")
        conn.execute(rule_sql)
        conn.commit()

        elapsed = time.time() - t0
        after_cnt = conn.execute("SELECT COUNT(*) FROM v8_cand_test").fetchone()[0]

        m = calculate_metrics(conn, sample_ids, gt, "v8_cand_test")
        delta_rec = (m['recall'] - base_metrics['recall']) * 100

        print(
            f"{name:<32} "
            f"{after_cnt - before_cnt:>12,} "
            f"{after_cnt:>12,} "
            f"{m['recall']*100:>9.4f}% "
            f"{delta_rec:>+10.4f}% "
            f"{m['avg']:>10.2f} "
            f"{elapsed:>7.2f}s"
        )

    conn.execute("DROP TABLE IF EXISTS temp.v8_cand_test")

    # ---------------------------------------------------------
    # PART B: SEQUENTIAL / CUMULATIVE ADDITION (V5 -> R17 -> R18 -> R19 -> R20)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("CUMULATIVE TEST: V5 -> R17 -> R18 -> R19 -> R20")
    print("=" * 80)
    print(f"{'CUMULATIVE STEP':<32} {'ADDED CAND':>12} {'TOTAL CAND':>12} {'RECALL':>10} {'DELTA REC':>11} {'AVG/S1':>10} {'TIME':>8}")
    print("-" * 100)

    print(
        f"{'V5 BASELINE':<32} "
        f"{'-':>12} "
        f"{base_metrics['total_candidates']:>12,} "
        f"{base_metrics['recall']*100:>9.4f}% "
        f"{'+0.0000%':>11} "
        f"{base_metrics['avg']:>10.2f} "
        f"{'-':>8}"
    )

    prev_recall = base_metrics['recall']

    for name, sql in new_candidate_rules:
        t0 = time.time()
        before_cnt = conn.execute("SELECT COUNT(*) FROM v8_candidates").fetchone()[0]

        conn.execute(sql)
        conn.commit()

        elapsed = time.time() - t0
        after_cnt = conn.execute("SELECT COUNT(*) FROM v8_candidates").fetchone()[0]

        m = calculate_metrics(conn, sample_ids, gt, "v8_candidates")
        delta_rec = (m['recall'] - prev_recall) * 100
        prev_recall = m['recall']

        print(
            f"{name:<32} "
            f"{after_cnt - before_cnt:>12,} "
            f"{after_cnt:>12,} "
            f"{m['recall']*100:>9.4f}% "
            f"{delta_rec:>+10.4f}% "
            f"{m['avg']:>10.2f} "
            f"{elapsed:>7.2f}s"
        )

    # ---------------------------------------------------------
    # FINAL METRICS AFTER ALL RULES
    # ---------------------------------------------------------
    final = calculate_metrics(conn, sample_ids, gt, "v8_candidates")

    no_match_rows = [
        sid for sid in sample_ids
        if not gt.get(sid, set())
    ]

    candidates = {}
    for sid, cid in conn.execute("SELECT source1_entity_id, candidate_entity_id FROM v8_candidates"):
        candidates.setdefault(sid, set()).add(cid)

    no_match_with_candidates = sum(
        bool(candidates.get(sid))
        for sid in no_match_rows
    )

    print("\n" + "=" * 80)
    print("FINAL V8 RESULTS (ALL 14 RULES COMBINED)")
    print("=" * 80)

    print(f"Candidate Pair Recall       : {final['recall']*100:.4f}%  (was {base_metrics['recall']*100:.4f}% in V5 baseline)")
    print(f"Candidate Pair Precision    : {final['precision']*100:.4f}%")
    print(f"All true matches covered    : {final['covered']}/{final['covered_total']} ({final['covered']/final['covered_total']*100:.2f}%)")
    print(f"Total candidate pairs       : {final['total_candidates']:,}")
    print(f"Average candidates / S1     : {final['avg']:.2f}  (was {base_metrics['avg']:.2f} in V5 baseline)")
    print(f"Median (P50) candidates     : {final['p50']:.0f}")
    print(f"P90 candidates              : {final['p90']:.0f}")
    print(f"P95 candidates              : {final['p95']:.0f}")
    print(f"P99 candidates              : {final['p99']:.0f}")
    print(f"Maximum candidates          : {final['max']:,}")
    print(f"Zero-candidate S1           : {final['zero']} (out of {len(sample_ids)})")
    print(f"No-match rows with cand.    : {no_match_with_candidates}/{len(no_match_rows)}")

    print("\nTemporary tables only -- original DB unchanged.")
    conn.close()


if __name__ == "__main__":
    main()
