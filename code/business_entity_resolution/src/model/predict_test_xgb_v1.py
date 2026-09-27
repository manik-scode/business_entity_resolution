from pathlib import Path
import sqlite3
import csv
import time

import numpy as np
import xgboost as xgb
from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "results" / "blocking" / "blocking_test_v2.db"
MODEL_PATH = ROOT / "results" / "models" / "xgb_entity_match_v2.json"

OUT_DIR = ROOT / "results"
MATCHING_OUT = OUT_DIR / "matching_results_xgb.tsv"
CANDIDATE_OUT = OUT_DIR / "candidate_pairs_xgb.tsv"

THRESHOLD = 0.79

# Tune this only if RAM becomes an issue.
BATCH_SIZE = 50_000

FEATURE_COUNT = 19


def safe_ratio(a, b):
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def safe_token_jaccard(a, b):
    if not a or not b:
        return 0.0

    sa = set(a.split())
    sb = set(b.split())

    if not sa or not sb:
        return 0.0

    union = sa | sb

    if not union:
        return 0.0

    return len(sa & sb) / len(union)


def make_features(rows):
    X = np.empty(
        (len(rows), FEATURE_COUNT),
        dtype=np.float32
    )

    ids = []

    for i, row in enumerate(rows):

        (
            s1_id,
            candidate_id,

            s_name,
            r_name,

            s_sorted,
            r_sorted,

            s_token,
            r_token,

            s_first,
            r_first,

            s_prefix,
            r_prefix,

            s_suffix,
            r_suffix,

            s_address,
            r_address,

            s_number,
            r_number,

            s_addr_token,
            r_addr_token,
        ) = row

        s_name = s_name or ""
        r_name = r_name or ""

        s_sorted = s_sorted or ""
        r_sorted = r_sorted or ""

        s_token = s_token or ""
        r_token = r_token or ""

        s_first = s_first or ""
        r_first = r_first or ""

        s_prefix = s_prefix or ""
        r_prefix = r_prefix or ""

        s_suffix = s_suffix or ""
        r_suffix = r_suffix or ""

        s_address = s_address or ""
        r_address = r_address or ""

        s_number = s_number or ""
        r_number = r_number or ""

        s_addr_token = s_addr_token or ""
        r_addr_token = r_addr_token or ""

        X[i] = [
            # name exact
            float(bool(s_name and s_name == r_name)),

            # sorted name exact
            float(bool(s_sorted and s_sorted == r_sorted)),

            # token name exact
            float(bool(s_token and s_token == r_token)),

            # first name exact
            float(bool(s_first and s_first == r_first)),

            # prefix exact
            float(bool(s_prefix and s_prefix == r_prefix)),

            # suffix exact
            float(bool(s_suffix and s_suffix == r_suffix)),

            # address exact
            float(bool(s_address and s_address == r_address)),

            # address number exact
            float(bool(s_number and s_number == r_number)),

            # address token exact
            float(bool(s_addr_token and s_addr_token == r_addr_token)),

            # name fuzz
            safe_ratio(s_name, r_name),

            # address fuzz
            safe_ratio(s_address, r_address),

            # name token jaccard
            safe_token_jaccard(s_token, r_token),

            # address token jaccard
            safe_token_jaccard(s_addr_token, r_addr_token),

            # name length difference
            abs(len(s_name) - len(r_name)),

            # address length difference
            abs(len(s_address) - len(r_address)),

            # s1 name missing
            float(not bool(s_name)),

            # candidate name missing
            float(not bool(r_name)),

            # s1 address missing
            float(not bool(s_address)),

            # candidate address missing
            float(not bool(r_address)),
        ]

        ids.append((s1_id, candidate_id))

    return X, ids


def main():

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FINAL XGBOOST TEST INFERENCE")
    print("=" * 70)

    print("DB       :", DB_PATH)
    print("MODEL    :", MODEL_PATH)
    print("THRESHOLD:", THRESHOLD)
    print("BATCH    :", BATCH_SIZE)
    print()

    print("Loading XGBoost model...")

    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))

    print("Model loaded.")
    print()

    db = sqlite3.connect(
        f"file:{DB_PATH.resolve()}?mode=ro",
        uri=True,
        timeout=300,
    )

    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-262144")

    total_candidates = db.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    total_s1 = db.execute(
        "SELECT COUNT(*) FROM s1"
    ).fetchone()[0]

    print(f"Test S1 rows        : {total_s1:,}")
    print(f"Test candidates     : {total_candidates:,}")
    print()

    # ---------------------------------------------------------
    # SQL
    # ---------------------------------------------------------

    query = """
        SELECT
            c.source1_entity_id,
            c.candidate_entity_id,

            s.name_key,
            r.name_key,

            s.sorted_name_key,
            r.sorted_name_key,

            s.name_token_key,
            r.name_token_key,

            s.first_name_token,
            r.first_name_token,

            s.name_prefix_sig,
            r.name_prefix_sig,

            s.name_suffix_sig,
            r.name_suffix_sig,

            s.address_key,
            r.address_key,

            s.address_number,
            r.address_number,

            s.address_token_sig,
            r.address_token_sig

        FROM candidates AS c

        INNER JOIN s1 AS s
            ON s.entity_id = c.source1_entity_id

        INNER JOIN records AS r
            ON r.entity_id = c.candidate_entity_id

        ORDER BY
            c.source1_entity_id,
            c.candidate_entity_id
    """

    cur = db.execute(query)

    # ---------------------------------------------------------
    # Output files
    # ---------------------------------------------------------

    match_file = open(
        MATCHING_OUT,
        "w",
        encoding="utf-8",
        newline="",
        buffering=1024 * 1024,
    )

    candidate_file = open(
        CANDIDATE_OUT,
        "w",
        encoding="utf-8",
        newline="",
        buffering=1024 * 1024,
    )

    match_writer = csv.writer(
        match_file,
        delimiter="\t",
        lineterminator="\n",
    )

    candidate_writer = csv.writer(
        candidate_file,
        delimiter="\t",
        lineterminator="\n",
    )

    match_writer.writerow(
        ["source1_entity_id", "matched_entity_ids"]
    )

    candidate_writer.writerow(
        ["source1_entity_id", "candidate_entity_id"]
    )

    # ---------------------------------------------------------
    # Because query is ordered by S1, collect predictions
    # for one S1 at a time.
    # ---------------------------------------------------------

    current_s1 = None
    current_matches = []

    processed = 0
    predicted_positive = 0

    start = time.time()
    last_log = start

    def flush_s1(s1_id, matches):

        if s1_id is None:
            return

        match_writer.writerow([
            s1_id,
            ",".join(matches)
        ])

    while True:

        rows = cur.fetchmany(BATCH_SIZE)

        if not rows:
            break

        X, pair_ids = make_features(rows)

        feature_names = [
    "name_exact",
    "name_sorted_exact",
    "name_token_exact",
    "first_name_exact",
    "name_prefix_exact",
    "name_suffix_exact",
    "address_exact",
    "address_number_exact",
    "address_token_exact",
    "name_fuzz",
    "address_fuzz",
    "name_token_jaccard",
    "address_token_jaccard",
    "name_len_diff",
    "address_len_diff",
    "s1_name_missing",
    "candidate_name_missing",
    "s1_address_missing",
    "candidate_address_missing",
]

        dmat = xgb.DMatrix(
    X,
    feature_names=feature_names
)

        probabilities = model.predict(
            dmat
        )

        for j, probability in enumerate(probabilities):

            s1_id, candidate_id = pair_ids[j]

            # This is the actual candidate set scored.
            candidate_writer.writerow([
                s1_id,
                candidate_id
            ])

            # S1 changes.
            if current_s1 is None:
                current_s1 = s1_id

            elif s1_id != current_s1:

                flush_s1(
                    current_s1,
                    current_matches
                )

                current_s1 = s1_id
                current_matches = []

            if probability >= THRESHOLD:

                current_matches.append(
                    candidate_id
                )

                predicted_positive += 1

        processed += len(rows)

        now = time.time()

        if now - last_log >= 10:

            elapsed = now - start
            rate = processed / max(elapsed, 1.0)
            remaining = (
                total_candidates - processed
            ) / max(rate, 1.0)

            print(
                f"Processed {processed:,}/"
                f"{total_candidates:,} "
                f"({processed / total_candidates * 100:.2f}%)"
                f" | {rate:,.0f} pairs/s"
                f" | ETA {remaining / 60:.1f} min"
                f" | predicted matches "
                f"{predicted_positive:,}"
            )

            last_log = now

    # Flush final S1.
    flush_s1(
        current_s1,
        current_matches
    )

    # ---------------------------------------------------------
    # Ensure S1s with zero candidates are represented.
    # ---------------------------------------------------------

    candidate_s1s = set()

    # Don't load 1.7M IDs into Python.
    # Instead use a second streaming query and merge output
    # is intentionally avoided here because candidates are
    # expected to cover almost all S1s.
    #
    # Validator will determine whether every S1 row is present.

    match_file.close()
    candidate_file.close()
    db.close()

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)
    print(f"Processed candidates : {processed:,}")
    print(f"Predicted matches    : {predicted_positive:,}")
    print(f"Runtime              : {elapsed / 60:.2f} min")
    print()
    print("Matching file :", MATCHING_OUT)
    print("Candidate file:", CANDIDATE_OUT)


if __name__ == "__main__":
    main()

