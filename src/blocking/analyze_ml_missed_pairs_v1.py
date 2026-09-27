"""
Analyze true matches that are missing from the ML candidate pool.

Goal:
    Find WHY the current blocking pipeline misses true pairs.

Current experiment:
    ML DB = results/ml/ml_train_v1.db
    Validation = same deterministic 20% S1 split used by XGBoost V2

The script:
1. Reproduces the validation S1 split.
2. Loads original ground truth.
3. Loads candidate pairs from ML DB.
4. Finds true pairs absent from candidates.
5. Loads blocking feature columns from blocking_train_v2.db.
6. Compares S1 and candidate blocking signals.
7. Reports which exact signals could recover the missed pairs.
8. Reports frequency of blocking keys.
"""

from __future__ import annotations

import csv
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

ML_DB = (
    ROOT
    / "results"
    / "ml"
    / "ml_train_v1.db"
)

BLOCKING_DB = (
    ROOT
    / "results"
    / "blocking"
    / "blocking_train_v2.db"
)

GROUND_TRUTH = (
    ROOT
    / "student_resource"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "blocking"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "ml_missed_pairs_analysis_v1.txt"
)

OUTPUT_PAIRS = (
    OUTPUT_DIR
    / "ml_missed_pairs_v1.tsv"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42
VALIDATION_FRACTION = 0.20


# ============================================================
# BLOCKING SIGNALS
# ============================================================

SIGNALS = [
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


# ============================================================
# HELPERS
# ============================================================

def normalize_value(value):
    """
    Convert NULL / empty values into None.
    """
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def reproduce_validation_split(s1_ids):
    """
    Reproduce the same deterministic validation split
    used by train_xgb_v2.py / evaluate_xgb_v2.py.
    """

    unique_ids = sorted(set(s1_ids))

    rng = np.random.default_rng(SEED)

    shuffled = np.array(
        unique_ids,
        dtype=object,
    )

    rng.shuffle(shuffled)

    validation_size = int(
        len(shuffled)
        * VALIDATION_FRACTION
    )

    return set(
        shuffled[:validation_size].tolist()
    )


# ============================================================
# LOAD VALIDATION S1 IDS
# ============================================================

def load_validation_s1_ids():
    print()
    print("=" * 70)
    print("LOADING ML S1 IDS")
    print("=" * 70)

    conn = sqlite3.connect(
        f"file:{ML_DB}?mode=ro",
        uri=True,
    )

    rows = conn.execute(
        """
        SELECT DISTINCT source1_entity_id
        FROM features
        """
    ).fetchall()

    conn.close()

    all_s1 = [
        row[0]
        for row in rows
    ]

    validation_s1 = reproduce_validation_split(
        all_s1
    )

    print(
        f"Total ML S1: "
        f"{len(all_s1):,}"
    )

    print(
        f"Validation S1: "
        f"{len(validation_s1):,}"
    )

    return validation_s1


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(validation_s1):
    print()
    print("=" * 70)
    print("LOADING GROUND TRUTH")
    print("=" * 70)

    ground_truth = {}

    with GROUND_TRUTH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            s1 = row[
                "source1_entity_id"
            ]

            if s1 not in validation_s1:
                continue

            raw = (
                row.get(
                    "matched_entity_ids"
                )
                or ""
            ).strip()

            if raw:
                matches = {
                    x.strip()
                    for x in raw.split(",")
                    if x.strip()
                }
            else:
                matches = set()

            ground_truth[s1] = matches

    print(
        f"Validation ground-truth S1: "
        f"{len(ground_truth):,}"
    )

    return ground_truth


# ============================================================
# LOAD CURRENT ML CANDIDATES
# ============================================================

def load_ml_candidates(validation_s1):
    print()
    print("=" * 70)
    print("LOADING CURRENT ML CANDIDATES")
    print("=" * 70)

    conn = sqlite3.connect(
        f"file:{ML_DB}?mode=ro",
        uri=True,
    )

    cursor = conn.execute(
        """
        SELECT
            source1_entity_id,
            candidate_entity_id
        FROM features
        """
    )

    candidate_pairs = set()

    for s1, candidate in cursor:

        if s1 not in validation_s1:
            continue

        candidate_pairs.add(
            (s1, candidate)
        )

    conn.close()

    print(
        f"Validation candidate pairs: "
        f"{len(candidate_pairs):,}"
    )

    return candidate_pairs


# ============================================================
# BUILD MISSED TRUE PAIRS
# ============================================================

def find_missed_pairs(
    validation_s1,
    ground_truth,
    candidate_pairs,
):
    print()
    print("=" * 70)
    print("FINDING MISSED TRUE PAIRS")
    print("=" * 70)

    missed = []

    total_true = 0
    covered = 0

    for s1 in validation_s1:

        true_matches = ground_truth.get(
            s1,
            set(),
        )

        for candidate in true_matches:

            total_true += 1

            pair = (
                s1,
                candidate,
            )

            if pair in candidate_pairs:
                covered += 1
            else:
                missed.append(pair)

    print(
        f"Total true pairs: "
        f"{total_true:,}"
    )

    print(
        f"Covered: "
        f"{covered:,}"
    )

    print(
        f"Missed: "
        f"{len(missed):,}"
    )

    print(
        f"Recall: "
        f"{covered / total_true:.4%}"
    )

    return missed


# ============================================================
# LOAD BLOCKING RECORDS
# ============================================================

def load_blocking_records(
    missed_pairs,
):
    """
    Load only the S1 and candidate records required
    to analyze the missed pairs.

    We use the existing blocking_train_v2.db.
    """

    print()
    print("=" * 70)
    print("LOADING BLOCKING RECORDS")
    print("=" * 70)

    needed_ids = set()

    for s1, candidate in missed_pairs:
        needed_ids.add(s1)
        needed_ids.add(candidate)

    print(
        f"Unique required entity IDs: "
        f"{len(needed_ids):,}"
    )

    conn = sqlite3.connect(
        f"file:{BLOCKING_DB}?mode=ro",
        uri=True,
    )

    records = {}

    cursor = conn.execute(
        """
        SELECT
            entity_id,
            source,
            country,
            name_key,
            sorted_name_key,
            name_token_key,
            first_name_token,
            address_key,
            address_number,
            name_prefix_sig,
            name_suffix_sig,
            address_token_sig
        FROM records
        """
    )

    checked = 0

    for row in cursor:

        entity_id = row[0]

        if entity_id not in needed_ids:
            continue

        records[entity_id] = {
            "source": row[1],
            "country": row[2],
            "name_key": normalize_value(row[3]),
            "sorted_name_key": normalize_value(row[4]),
            "name_token_key": normalize_value(row[5]),
            "first_name_token": normalize_value(row[6]),
            "address_key": normalize_value(row[7]),
            "address_number": normalize_value(row[8]),
            "name_prefix_sig": normalize_value(row[9]),
            "name_suffix_sig": normalize_value(row[10]),
            "address_token_sig": normalize_value(row[11]),
        }

        checked += 1

        if checked % 1_000_000 == 0:
            print(
                f"Loaded required records: "
                f"{checked:,}"
            )

    conn.close()

    print(
        f"Records loaded: "
        f"{len(records):,}"
    )

    missing_records = (
        needed_ids
        - set(records.keys())
    )

    if missing_records:

        print(
            f"WARNING: "
            f"{len(missing_records):,} "
            f"required records were not found."
        )

    return records


# ============================================================
# SIGNAL ANALYSIS
# ============================================================

def analyze_signals(
    missed_pairs,
    records,
):
    print()
    print("=" * 70)
    print("ANALYZING BLOCKING SIGNALS")
    print("=" * 70)

    signal_counts = Counter()

    combined_counts = Counter()

    country_mismatch = 0

    analyzable = 0

    examples = defaultdict(list)

    for s1, candidate in missed_pairs:

        s1_record = records.get(s1)
        candidate_record = records.get(candidate)

        if not s1_record or not candidate_record:
            continue

        analyzable += 1

        if (
            s1_record["country"]
            != candidate_record["country"]
        ):
            country_mismatch += 1

        matched_signals = []

        for signal in SIGNALS:

            s1_value = s1_record[signal]
            candidate_value = (
                candidate_record[signal]
            )

            if (
                s1_value is not None
                and candidate_value is not None
                and s1_value == candidate_value
            ):

                signal_counts[signal] += 1
                matched_signals.append(signal)

        if matched_signals:

            for signal in matched_signals:
                if len(examples[signal]) < 5:
                    examples[signal].append(
                        (
                            s1,
                            candidate,
                        )
                    )

        if matched_signals:

            combo = tuple(
                sorted(matched_signals)
            )

            combined_counts[combo] += 1

    print(
        f"Analyzable missed pairs: "
        f"{analyzable:,}"
    )

    print(
        f"Country mismatch: "
        f"{country_mismatch:,}"
    )

    print()

    print(
        "Individual signal coverage:"
    )

    print(
        "-" * 70
    )

    for signal in SIGNALS:

        count = signal_counts[signal]

        pct = (
            count / analyzable
            if analyzable
            else 0
        )

        print(
            f"{signal:25s} "
            f"{count:8,} "
            f"{pct:8.2%}"
        )

    any_signal = 0

    for combo, count in combined_counts.items():

        if combo:
            any_signal += count

    no_signal = (
        analyzable
        - any_signal
    )

    print()
    print(
        f"Any exact signal: "
        f"{any_signal:,} "
        f"({any_signal / analyzable:.2%})"
    )

    print(
        f"No exact signal: "
        f"{no_signal:,} "
        f"({no_signal / analyzable:.2%})"
    )

    print()

    print(
        "Most common signal combinations:"
    )

    print(
        "-" * 70
    )

    for combo, count in combined_counts.most_common(30):

        label = (
            " + ".join(combo)
            if combo
            else "NO_SIGNAL"
        )

        print(
            f"{count:8,}  {label}"
        )

    return {
        "signal_counts": signal_counts,
        "combined_counts": combined_counts,
        "analyzable": analyzable,
        "country_mismatch": country_mismatch,
        "any_signal": any_signal,
        "no_signal": no_signal,
        "examples": examples,
    }


# ============================================================
# RULE RECOVERY ANALYSIS
# ============================================================

def analyze_rule_recovery(
    missed_pairs,
    records,
):
    """
    Simulate simple exact blocking rules on missed pairs.

    This DOES NOT generate the full candidate DB.

    It only asks:
        "If we used this exact key combination,
         how many currently missed true pairs would
         become candidates?"
    """

    print()
    print("=" * 70)
    print("RULE RECOVERY ANALYSIS")
    print("=" * 70)

    rules = {
        "R_name_exact": [
            "name_key",
        ],

        "R_sorted_name": [
            "sorted_name_key",
        ],

        "R_name_token": [
            "name_token_key",
        ],

        "R_first_name": [
            "first_name_token",
        ],

        "R_address_exact": [
            "address_key",
        ],

        "R_address_number": [
            "address_number",
        ],

        "R_name_prefix": [
            "name_prefix_sig",
        ],

        "R_name_suffix": [
            "name_suffix_sig",
        ],

        "R_address_token": [
            "address_token_sig",
        ],

        "R_number_plus_address_token": [
            "address_number",
            "address_token_sig",
        ],

        "R_first_name_plus_address_token": [
            "first_name_token",
            "address_token_sig",
        ],

        "R_prefix_plus_number": [
            "name_prefix_sig",
            "address_number",
        ],

        "R_suffix_plus_number": [
            "name_suffix_sig",
            "address_number",
        ],

        "R_name_token_plus_address_token": [
            "name_token_key",
            "address_token_sig",
        ],
    }

    recovery = {}

    for rule_name, fields in rules.items():

        recovered = 0

        for s1, candidate in missed_pairs:

            s1_record = records.get(s1)
            candidate_record = records.get(candidate)

            if (
                not s1_record
                or not candidate_record
            ):
                continue

            # Country should always match for a valid
            # cross-source entity resolution pair.
            if (
                s1_record["country"]
                != candidate_record["country"]
            ):
                continue

            matched = True

            for field in fields:

                a = s1_record[field]
                b = candidate_record[field]

                if (
                    a is None
                    or b is None
                    or a != b
                ):
                    matched = False
                    break

            if matched:
                recovered += 1

        recovery[rule_name] = recovered

    print()

    for rule_name, recovered in sorted(
        recovery.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        pct = (
            recovered / len(missed_pairs)
            if missed_pairs
            else 0
        )

        print(
            f"{rule_name:40s} "
            f"{recovered:8,} "
            f"{pct:8.2%}"
        )

    return recovery


# ============================================================
# WRITE MISSED PAIRS
# ============================================================

def save_missed_pairs(
    missed_pairs,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PAIRS.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f,
            delimiter="\t",
        )

        writer.writerow(
            [
                "source1_entity_id",
                "true_candidate_entity_id",
            ]
        )

        for s1, candidate in missed_pairs:

            writer.writerow(
                [
                    s1,
                    candidate,
                ]
            )

    print()
    print(
        f"Missed pairs saved: "
        f"{OUTPUT_PAIRS}"
    )


# ============================================================
# WRITE REPORT
# ============================================================

def save_report(
    missed_pairs,
    signal_analysis,
    rule_recovery,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_REPORT.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "ML MISSED TRUE PAIR ANALYSIS V1\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            f"Missed true pairs: "
            f"{len(missed_pairs):,}\n"
        )

        f.write(
            f"Analyzable pairs: "
            f"{signal_analysis['analyzable']:,}\n"
        )

        f.write(
            f"Any exact signal: "
            f"{signal_analysis['any_signal']:,}\n"
        )

        f.write(
            f"No exact signal: "
            f"{signal_analysis['no_signal']:,}\n\n"
        )

        f.write(
            "INDIVIDUAL SIGNALS\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        for signal in SIGNALS:

            count = signal_analysis[
                "signal_counts"
            ][signal]

            pct = (
                count
                / signal_analysis["analyzable"]
                if signal_analysis["analyzable"]
                else 0
            )

            f.write(
                f"{signal:30s} "
                f"{count:10,} "
                f"{pct:8.2%}\n"
            )

        f.write(
            "\nRULE RECOVERY\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        for rule, count in sorted(
            rule_recovery.items(),
            key=lambda x: x[1],
            reverse=True,
        ):

            pct = (
                count / len(missed_pairs)
                if missed_pairs
                else 0
            )

            f.write(
                f"{rule:40s} "
                f"{count:10,} "
                f"{pct:8.2%}\n"
            )

        f.write(
            "\nSIGNAL COMBINATIONS\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        for combo, count in (
            signal_analysis[
                "combined_counts"
            ].most_common(50)
        ):

            label = (
                " + ".join(combo)
                if combo
                else "NO_SIGNAL"
            )

            f.write(
                f"{count:10,}  {label}\n"
            )

    print()
    print(
        f"Report saved: "
        f"{OUTPUT_REPORT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ML MISSED PAIRS — BLOCKING FORENSIC ANALYSIS")
    print("=" * 70)

    print()
    print(
        "This analyzes only the current 100k-S1 "
        "ML experiment."
    )

    # --------------------------------------------------------
    # 1. Validation S1s
    # --------------------------------------------------------

    validation_s1 = (
        load_validation_s1_ids()
    )

    # --------------------------------------------------------
    # 2. Ground truth
    # --------------------------------------------------------

    ground_truth = load_ground_truth(
        validation_s1
    )

    # --------------------------------------------------------
    # 3. Current ML candidates
    # --------------------------------------------------------

    candidate_pairs = (
        load_ml_candidates(
            validation_s1
        )
    )

    # --------------------------------------------------------
    # 4. Find missed true pairs
    # --------------------------------------------------------

    missed_pairs = find_missed_pairs(
        validation_s1,
        ground_truth,
        candidate_pairs,
    )

    if not missed_pairs:

        print()
        print(
            "No missed pairs found."
        )
        return

    # --------------------------------------------------------
    # 5. Save missed pairs
    # --------------------------------------------------------

    save_missed_pairs(
        missed_pairs
    )

    # --------------------------------------------------------
    # 6. Load blocking records
    # --------------------------------------------------------

    records = load_blocking_records(
        missed_pairs
    )

    # --------------------------------------------------------
    # 7. Analyze exact signals
    # --------------------------------------------------------

    signal_analysis = analyze_signals(
        missed_pairs,
        records,
    )

    # --------------------------------------------------------
    # 8. Analyze possible rules
    # --------------------------------------------------------

    rule_recovery = analyze_rule_recovery(
        missed_pairs,
        records,
    )

    # --------------------------------------------------------
    # 9. Save report
    # --------------------------------------------------------

    save_report(
        missed_pairs,
        signal_analysis,
        rule_recovery,
    )

    # --------------------------------------------------------
    # 10. Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FORENSIC ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Missed true pairs: "
        f"{len(missed_pairs):,}"
    )

    print(
        f"Pairs with at least one exact signal: "
        f"{signal_analysis['any_signal']:,}"
    )

    print(
        f"Pairs with NO exact signal: "
        f"{signal_analysis['no_signal']:,}"
    )

    print()
    print(
        "Do NOT add a new blocking rule yet."
    )

    print(
        "First inspect the rule-recovery numbers "
        "above; then we choose the smallest "
        "high-value blocking expansion."
    )


if __name__ == "__main__":
    main()