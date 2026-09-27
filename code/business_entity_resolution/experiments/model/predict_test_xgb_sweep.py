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

# New thresholds only.
# Already tested: 0.80, 0.85, 0.90, 0.93, 0.95
THRESHOLDS = {
    "079": 0.79,
    "081": 0.81,
    "082": 0.82,
    "083": 0.83,
    "084": 0.84,
    "086": 0.86,
    "087": 0.87,
    "088": 0.88,
    "089": 0.89,
    "091": 0.91,
    "092": 0.92,
    "094": 0.94,
}

BATCH_SIZE = 50_000
FEATURE_COUNT = 19


FEATURE_NAMES = [
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
            float(bool(s_name and s_name == r_name)),
            float(bool(s_sorted and s_sorted == r_sorted)),
            float(bool(s_token and s_token == r_token)),
            float(bool(s_first and s_first == r_first)),
            float(bool(s_prefix and s_prefix == r_prefix)),
            float(bool(s_suffix and s_suffix == r_suffix)),
            float(bool(s_address and s_address == r_address)),
            float(bool(s_number and s_number == r_number)),
            float(bool(s_addr_token and s_addr_token == r_addr_token)),
            safe_ratio(s_name, r_name),
            safe_ratio(s_address, r_address),
            safe_token_jaccard(s_token, r_token),
            safe_token_jaccard(s_addr_token, r_addr_token),
            abs(len(s_name) - len(r_name)),
            abs(len(s_address) - len(r_address)),
            float(not bool(s_name)),
            float(not bool(r_name)),
            float(not bool(s_address)),
            float(not bool(r_address)),
        ]

        ids.append((s1_id, candidate_id))

    return X, ids


def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 75)
    print("XGBOOST THRESHOLD SWEEP")
    print("=" * 75)

    print("DB        :", DB_PATH)
    print("MODEL     :", MODEL_PATH)
    print(
        "Thresholds:",
        [f"{v:.2f}" for v in THRESHOLDS.values()]
    )
    print("Batch     :", BATCH_SIZE)
    print()

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    print("Loading XGBoost model...")

    model = xgb.Booster()
    model.load_model(str(MODEL_PATH))

    print("Model loaded.")
    print()

    # ---------------------------------------------------------
    # Open DB
    # ---------------------------------------------------------

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

    candidate_s1 = db.execute(
        "SELECT COUNT(DISTINCT source1_entity_id) FROM candidates"
    ).fetchone()[0]

    print(f"Test S1 rows       : {total_s1:,}")
    print(f"S1 with candidates : {candidate_s1:,}")
    print(
        f"Zero-candidate S1  : "
        f"{total_s1 - candidate_s1:,}"
    )
    print(f"Test candidates    : {total_candidates:,}")
    print()

    # ---------------------------------------------------------
    # Output files
    # ---------------------------------------------------------

    writers = {}
    files = {}

    for suffix in THRESHOLDS:

        path = (
            OUT_DIR /
            f"matching_results_xgb_t{suffix}.tsv"
        )

        f = open(
            path,
            "w",
            encoding="utf-8",
            newline="",
            buffering=1024 * 1024,
        )

        writer = csv.writer(
            f,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writerow([
            "source1_entity_id",
            "matched_entity_ids"
        ])

        files[suffix] = f
        writers[suffix] = writer

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

    current_s1 = None

    matches = {
        suffix: []
        for suffix in THRESHOLDS
    }

    predicted_counts = {
        suffix: 0
        for suffix in THRESHOLDS
    }

    processed = 0

    start = time.time()
    last_log = start

    # ---------------------------------------------------------
    # Flush one S1
    # ---------------------------------------------------------

    def flush_s1(s1_id):

        if s1_id is None:
            return

        for suffix in THRESHOLDS:

            writers[suffix].writerow([
                s1_id,
                ",".join(matches[suffix])
            ])

            matches[suffix].clear()

    # ---------------------------------------------------------
    # Main inference
    # ---------------------------------------------------------

    while True:

        rows = cur.fetchmany(BATCH_SIZE)

        if not rows:
            break

        X, pair_ids = make_features(rows)

        dmat = xgb.DMatrix(
            X,
            feature_names=FEATURE_NAMES
        )

        probabilities = model.predict(dmat)

        for j, probability in enumerate(probabilities):

            s1_id, candidate_id = pair_ids[j]

            if current_s1 is None:

                current_s1 = s1_id

            elif s1_id != current_s1:

                flush_s1(current_s1)

                current_s1 = s1_id

            for suffix, threshold in THRESHOLDS.items():

                if probability >= threshold:

                    matches[suffix].append(
                        candidate_id
                    )

                    predicted_counts[suffix] += 1

        processed += len(rows)

        now = time.time()

        if now - last_log >= 10:

            elapsed = now - start

            rate = processed / max(
                elapsed,
                1.0
            )

            remaining = (
                total_candidates - processed
            ) / max(rate, 1.0)

            progress = (
                processed /
                total_candidates *
                100
            )

            counts_text = " | ".join(
                f"t{suffix}={predicted_counts[suffix]:,}"
                for suffix in THRESHOLDS
            )

            print(
                f"Processed {processed:,}/"
                f"{total_candidates:,} "
                f"({progress:.2f}%)"
                f" | {rate:,.0f} pairs/s"
                f" | ETA {remaining / 60:.1f} min"
                f" | {counts_text}"
            )

            last_log = now

    # ---------------------------------------------------------
    # Flush final candidate-bearing S1
    # ---------------------------------------------------------

    flush_s1(current_s1)

    db.close()

    for f in files.values():
        f.close()

    elapsed = time.time() - start

    print()
    print("=" * 75)
    print("INFERENCE COMPLETE")
    print("=" * 75)

    print(
        f"Processed candidates : "
        f"{processed:,}"
    )

    print(
        f"Runtime              : "
        f"{elapsed / 60:.2f} min"
    )

    print()

    for suffix, threshold in THRESHOLDS.items():

        print(
            f"Threshold {threshold:.2f}"
            f" -> predicted matches: "
            f"{predicted_counts[suffix]:,}"
        )

    print()

    # ---------------------------------------------------------
    # Add zero-candidate S1 rows
    # ---------------------------------------------------------

    print("Adding zero-candidate S1 rows...")

    for suffix in THRESHOLDS:

        partial_path = (
            OUT_DIR /
            f"matching_results_xgb_t{suffix}.tsv"
        )

        final_path = (
            OUT_DIR /
            f"matching_results_xgb_t{suffix}_final.tsv"
        )

        temp_path = (
            OUT_DIR /
            f"matching_results_xgb_t{suffix}_partial.tsv"
        )

        partial_path.rename(temp_path)

        with open(
            temp_path,
            "r",
            encoding="utf-8",
            newline=""
        ) as src, open(
            final_path,
            "w",
            encoding="utf-8",
            newline="",
            buffering=1024 * 1024,
        ) as dst:

            reader = csv.reader(
                src,
                delimiter="\t"
            )

            writer = csv.writer(
                dst,
                delimiter="\t",
                lineterminator="\n"
            )

            next(reader)

            writer.writerow([
                "source1_entity_id",
                "matched_entity_ids"
            ])

            partial_row = next(
                reader,
                None
            )

            s1_db = sqlite3.connect(
                f"file:{DB_PATH.resolve()}?mode=ro",
                uri=True,
                timeout=300,
            )

            s1_db.execute(
                "PRAGMA query_only=ON"
            )

            all_s1 = s1_db.execute(
                "SELECT entity_id "
                "FROM s1 "
                "ORDER BY entity_id"
            )

            rows_written = 0

            for (s1_id,) in all_s1:

                if (
                    partial_row is not None
                    and partial_row[0] == s1_id
                ):

                    writer.writerow(
                        partial_row
                    )

                    partial_row = next(
                        reader,
                        None
                    )

                else:

                    writer.writerow([
                        s1_id,
                        ""
                    ])

                rows_written += 1

            s1_db.close()

        Path(temp_path).unlink()

        print(
            f"Created: {final_path.name} "
            f"({rows_written:,} S1 rows)"
        )

    print()
    print("=" * 75)
    print("ALL NEW THRESHOLD FILES READY")
    print("=" * 75)

    for suffix in THRESHOLDS:

        print(
            OUT_DIR /
            f"matching_results_xgb_t{suffix}_final.tsv"
        )


if __name__ == "__main__":
    main()