"""
Evaluate XGBoost entity-resolution model at S1/entity level.

Purpose:
- Reproduce the exact 80/20 S1-level validation split used during training.
- Run predictions on validation candidate pairs.
- Compare predictions against the ORIGINAL train ground truth.
- Evaluate multiple probability thresholds using S1-level F0.5.
- Measure candidate coverage ceiling separately from model performance.

Important:
- This evaluates ONLY the 100k-S1 ML experiment.
- It does NOT run inference on the full test set.
- It does NOT modify the source blocking DB.
- It does NOT overwrite the existing pair-level threshold file.

Run from repository root:

    python src\model\evaluate_xgb_v2.py
"""

from __future__ import annotations

import csv
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

ML_DB = ROOT / "results" / "ml" / "ml_train_v1.db"

MODEL_PATH = ROOT / "results" / "models" / "xgb_entity_match_v2.json"

GROUND_TRUTH_PATH = (
    ROOT
    / "student_resource"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

OUTPUT_DIR = ROOT / "results" / "models"

OUTPUT_THRESHOLD_PATH = (
    OUTPUT_DIR / "xgb_entity_match_v2_s1_threshold.txt"
)

OUTPUT_REPORT_PATH = (
    OUTPUT_DIR / "xgb_entity_match_v2_s1_evaluation.txt"
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

VALIDATION_FRACTION = 0.20

# These are the thresholds we want to compare.
THRESHOLDS = [
    0.80,
    0.82,
    0.84,
    0.85,
    0.86,
    0.88,
    0.90,
    0.91,
    0.92,
    0.93,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
]

# Exact feature order used by train_xgb_v2.py.
FEATURE_COLUMNS = [
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


# ============================================================
# HELPERS
# ============================================================

def f05_score(precision: float, recall: float) -> float:
    """
    F0.5 gives twice as much weight to precision as recall.
    """
    if precision == 0.0 and recall == 0.0:
        return 0.0

    beta_squared = 0.5 ** 2

    denominator = (
        beta_squared * precision + recall
    )

    if denominator == 0.0:
        return 0.0

    return (
        (1.0 + beta_squared)
        * precision
        * recall
        / denominator
    )


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_s1_metrics(
    true_ids: set[str],
    predicted_ids: set[str],
) -> tuple[float, float, float, int]:
    """
    Calculate precision, recall and F0.5 for one S1.

    Competition-style empty behavior:
        true empty + predicted empty => perfect 1.0

    Other empty cases:
        true empty + prediction non-empty => 0
        true non-empty + prediction empty => 0
    """

    true_count = len(true_ids)
    predicted_count = len(predicted_ids)

    if true_count == 0 and predicted_count == 0:
        return 1.0, 1.0, 1.0, 0

    if true_count == 0 and predicted_count > 0:
        return 0.0, 0.0, 0.0, 0

    if true_count > 0 and predicted_count == 0:
        return 0.0, 0.0, 0.0, 0

    correct = len(true_ids.intersection(predicted_ids))

    precision = correct / predicted_count
    recall = correct / true_count

    f05 = f05_score(precision, recall)

    return precision, recall, f05, correct


def load_ground_truth(
    path: Path,
    validation_s1_ids: set[str],
) -> dict[str, set[str]]:
    """
    Load only ground-truth rows belonging to validation S1 IDs.
    """

    print()
    print("=" * 70)
    print("LOADING GROUND TRUTH")
    print("=" * 70)

    ground_truth: dict[str, set[str]] = {}

    rows_read = 0
    rows_used = 0

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:
            rows_read += 1

            s1_id = row["source1_entity_id"]

            if s1_id not in validation_s1_ids:
                continue

            raw_matches = (
                row.get("matched_entity_ids") or ""
            ).strip()

            if raw_matches:
                matches = {
                    x.strip()
                    for x in raw_matches.split(",")
                    if x.strip()
                }
            else:
                matches = set()

            ground_truth[s1_id] = matches
            rows_used += 1

    print(f"Ground-truth rows scanned : {rows_read:,}")
    print(f"Validation S1 rows used   : {rows_used:,}")

    return ground_truth


def reproduce_validation_split(
    s1_ids: list[str],
) -> set[str]:
    """
    Reproduce the same S1-level 80/20 split.

    IMPORTANT:
    This must match the split logic in train_xgb_v2.py.

    We sort IDs first to make the split deterministic regardless
    of SQLite row-return order.
    """

    unique_ids = sorted(set(s1_ids))

    rng = np.random.default_rng(RANDOM_SEED)

    shuffled = np.array(
        unique_ids,
        dtype=object,
    )

    rng.shuffle(shuffled)

    validation_size = int(
        len(shuffled) * VALIDATION_FRACTION
    )

    validation_ids = set(
        shuffled[:validation_size].tolist()
    )

    return validation_ids


# ============================================================
# LOAD ML DATA
# ============================================================

def load_ml_data():
    print()
    print("=" * 70)
    print("LOADING ML DATA")
    print("=" * 70)

    if not ML_DB.exists():
        raise FileNotFoundError(
            f"ML database not found:\n{ML_DB}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found:\n{MODEL_PATH}"
        )

    conn = sqlite3.connect(
        f"file:{ML_DB}?mode=ro",
        uri=True,
    )

    columns_sql = ", ".join(
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
        + FEATURE_COLUMNS
        + ["label"]
    )

    query = f"""
        SELECT
            {columns_sql}
        FROM features
    """

    cursor = conn.execute(query)

    rows = cursor.fetchall()

    conn.close()

    print(f"ML rows loaded: {len(rows):,}")

    if not rows:
        raise RuntimeError(
            "No rows found in ML feature database."
        )

    return rows


# ============================================================
# BUILD MATRICES
# ============================================================

def prepare_validation_data(rows):
    print()
    print("=" * 70)
    print("REPRODUCING VALIDATION SPLIT")
    print("=" * 70)

    s1_ids = [
        row[0]
        for row in rows
    ]

    validation_s1_ids = reproduce_validation_split(
        s1_ids
    )

    print(
        f"Unique S1 IDs: "
        f"{len(set(s1_ids)):,}"
    )

    print(
        f"Validation S1 IDs: "
        f"{len(validation_s1_ids):,}"
    )

    feature_start = 2

    feature_end = (
        feature_start
        + len(FEATURE_COLUMNS)
    )

    validation_rows = [
        row
        for row in rows
        if row[0] in validation_s1_ids
    ]

    print(
        f"Validation candidate rows: "
        f"{len(validation_rows):,}"
    )

    X = np.asarray(
        [
            row[
                feature_start:feature_end
            ]
            for row in validation_rows
        ],
        dtype=np.float32,
    )

    y = np.asarray(
        [
            row[feature_end]
            for row in validation_rows
        ],
        dtype=np.int8,
    )

    candidate_s1 = [
        row[0]
        for row in validation_rows
    ]

    candidate_ids = [
        row[1]
        for row in validation_rows
    ]

    print(
        f"Feature matrix shape: "
        f"{X.shape}"
    )

    print(
        f"Validation positives: "
        f"{int(y.sum()):,}"
    )

    print(
        f"Validation negatives: "
        f"{int((y == 0).sum()):,}"
    )

    return (
        validation_s1_ids,
        validation_rows,
        X,
        y,
        candidate_s1,
        candidate_ids,
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict(model_path: Path, X: np.ndarray) -> np.ndarray:
    print()
    print("=" * 70)
    print("RUNNING XGBOOST PREDICTIONS")
    print("=" * 70)

    print(f"Model: {model_path}")

    model = xgb.Booster()

    model.load_model(
        str(model_path)
    )

    dmatrix = xgb.DMatrix(
        X,
        feature_names=FEATURE_COLUMNS,
    )

    probabilities = model.predict(
        dmatrix
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    print(
        f"Predictions: "
        f"{len(probabilities):,}"
    )

    print(
        f"Probability min: "
        f"{probabilities.min():.6f}"
    )

    print(
        f"Probability max: "
        f"{probabilities.max():.6f}"
    )

    print(
        f"Probability mean: "
        f"{probabilities.mean():.6f}"
    )

    return probabilities


# ============================================================
# CANDIDATE CEILING
# ============================================================

def calculate_candidate_ceiling(
    validation_s1_ids: set[str],
    candidate_s1: list[str],
    candidate_ids: list[str],
    ground_truth: dict[str, set[str]],
):
    """
    Measure how much of the actual ground truth is present
    inside the candidate pool.

    This is the maximum possible performance the ML model
    can achieve without changing blocking.
    """

    print()
    print("=" * 70)
    print("CANDIDATE COVERAGE CEILING")
    print("=" * 70)

    candidate_sets: dict[str, set[str]] = defaultdict(set)

    for s1, candidate in zip(
        candidate_s1,
        candidate_ids,
    ):
        candidate_sets[s1].add(candidate)

    total_true_pairs = 0
    covered_true_pairs = 0

    s1_with_true = 0
    s1_fully_covered = 0

    s1_coverage_values = []

    for s1 in validation_s1_ids:
        true_ids = ground_truth.get(
            s1,
            set(),
        )

        candidate_ids_for_s1 = candidate_sets.get(
            s1,
            set(),
        )

        if true_ids:
            s1_with_true += 1

            covered = len(
                true_ids.intersection(
                    candidate_ids_for_s1
                )
            )

            total_true_pairs += len(true_ids)
            covered_true_pairs += covered

            coverage = (
                covered / len(true_ids)
            )

            s1_coverage_values.append(
                coverage
            )

            if covered == len(true_ids):
                s1_fully_covered += 1

    pair_recall = safe_div(
        covered_true_pairs,
        total_true_pairs,
    )

    mean_s1_coverage = (
        float(np.mean(s1_coverage_values))
        if s1_coverage_values
        else 0.0
    )

    print(
        f"S1s with >=1 true match : "
        f"{s1_with_true:,}"
    )

    print(
        f"Total true pairs        : "
        f"{total_true_pairs:,}"
    )

    print(
        f"True pairs in candidates : "
        f"{covered_true_pairs:,}"
    )

    print(
        f"Candidate pair recall   : "
        f"{pair_recall:.4%}"
    )

    print(
        f"S1 fully covered        : "
        f"{s1_fully_covered:,} / "
        f"{s1_with_true:,} "
        f"("
        f"{safe_div(s1_fully_covered, s1_with_true):.4%}"
        f")"
    )

    print(
        f"Mean S1 candidate recall: "
        f"{mean_s1_coverage:.4%}"
    )

    return {
        "total_true_pairs": total_true_pairs,
        "covered_true_pairs": covered_true_pairs,
        "candidate_pair_recall": pair_recall,
        "s1_with_true": s1_with_true,
        "s1_fully_covered": s1_fully_covered,
        "mean_s1_coverage": mean_s1_coverage,
    }


# ============================================================
# S1-LEVEL THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    threshold: float,
    validation_s1_ids: set[str],
    candidate_s1: list[str],
    candidate_ids: list[str],
    probabilities: np.ndarray,
    ground_truth: dict[str, set[str]],
):
    """
    Evaluate one probability threshold at S1 level.
    """

    predictions: dict[str, set[str]] = defaultdict(set)

    for s1, candidate_id, probability in zip(
        candidate_s1,
        candidate_ids,
        probabilities,
    ):

        if probability >= threshold:
            predictions[s1].add(
                candidate_id
            )

    s1_precisions = []
    s1_recalls = []
    s1_f05_scores = []

    total_true = 0
    total_predicted = 0
    total_correct = 0

    empty_true = 0
    empty_predicted = 0

    correct_s1 = 0

    exact_set_match = 0

    prediction_counts = []
    true_counts = []

    for s1 in validation_s1_ids:

        true_ids = ground_truth.get(
            s1,
            set(),
        )

        predicted_ids = predictions.get(
            s1,
            set(),
        )

        precision, recall, f05, correct = (
            calculate_s1_metrics(
                true_ids,
                predicted_ids,
            )
        )

        s1_precisions.append(
            precision
        )

        s1_recalls.append(
            recall
        )

        s1_f05_scores.append(
            f05
        )

        total_true += len(true_ids)
        total_predicted += len(predicted_ids)
        total_correct += correct

        prediction_counts.append(
            len(predicted_ids)
        )

        true_counts.append(
            len(true_ids)
        )

        if not true_ids:
            empty_true += 1

        if not predicted_ids:
            empty_predicted += 1

        if correct == len(true_ids) and (
            len(predicted_ids)
            == len(true_ids)
        ):
            exact_set_match += 1

        if f05 == 1.0:
            correct_s1 += 1

    macro_precision = float(
        np.mean(s1_precisions)
    )

    macro_recall = float(
        np.mean(s1_recalls)
    )

    macro_f05 = float(
        np.mean(s1_f05_scores)
    )

    micro_precision = safe_div(
        total_correct,
        total_predicted,
    )

    micro_recall = safe_div(
        total_correct,
        total_true,
    )

    micro_f05 = f05_score(
        micro_precision,
        micro_recall,
    )

    mean_predicted = float(
        np.mean(prediction_counts)
    )

    median_predicted = float(
        np.median(prediction_counts)
    )

    p90_predicted = float(
        np.percentile(
            prediction_counts,
            90,
        )
    )

    max_predicted = int(
        max(prediction_counts)
    ) if prediction_counts else 0

    mean_true = float(
        np.mean(true_counts)
    )

    return {
        "threshold": threshold,

        # Competition-style macro metrics.
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f05": macro_f05,

        # Pair/micro metrics.
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f05": micro_f05,

        # Counts.
        "total_true": total_true,
        "total_predicted": total_predicted,
        "total_correct": total_correct,

        # Empty behavior.
        "empty_true": empty_true,
        "empty_predicted": empty_predicted,

        # Exact S1 behavior.
        "perfect_s1": correct_s1,
        "exact_set_match_s1": exact_set_match,

        # Prediction size.
        "mean_predicted": mean_predicted,
        "median_predicted": median_predicted,
        "p90_predicted": p90_predicted,
        "max_predicted": max_predicted,

        "mean_true": mean_true,
    }


# ============================================================
# REPORT
# ============================================================

def print_results(
    results: list[dict],
    candidate_ceiling: dict,
    validation_s1_count: int,
):
    print()
    print("=" * 110)
    print("S1-LEVEL THRESHOLD RESULTS")
    print("=" * 110)

    header = (
        f"{'Thr':>6} "
        f"{'Macro P':>10} "
        f"{'Macro R':>10} "
        f"{'Macro F0.5':>12} "
        f"{'Micro P':>10} "
        f"{'Micro R':>10} "
        f"{'Pred/S1':>10} "
        f"{'Exact S1':>10}"
    )

    print(header)
    print("-" * 110)

    for result in results:

        print(
            f"{result['threshold']:>6.2f} "
            f"{result['macro_precision']:>10.4f} "
            f"{result['macro_recall']:>10.4f} "
            f"{result['macro_f05']:>12.4f} "
            f"{result['micro_precision']:>10.4f} "
            f"{result['micro_recall']:>10.4f} "
            f"{result['mean_predicted']:>10.2f} "
            f"{result['exact_set_match_s1']:>10,}"
        )

    best = max(
        results,
        key=lambda x: (
            x["macro_f05"],
            x["macro_precision"],
            x["macro_recall"],
        ),
    )

    print()
    print("=" * 70)
    print("BEST S1-LEVEL THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold           : "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Macro Precision     : "
        f"{best['macro_precision']:.6f}"
    )

    print(
        f"Macro Recall        : "
        f"{best['macro_recall']:.6f}"
    )

    print(
        f"Macro F0.5          : "
        f"{best['macro_f05']:.6f}"
    )

    print(
        f"Micro Precision     : "
        f"{best['micro_precision']:.6f}"
    )

    print(
        f"Micro Recall        : "
        f"{best['micro_recall']:.6f}"
    )

    print(
        f"Micro F0.5          : "
        f"{best['micro_f05']:.6f}"
    )

    print(
        f"Mean predicted/S1   : "
        f"{best['mean_predicted']:.3f}"
    )

    print(
        f"Median predicted/S1 : "
        f"{best['median_predicted']:.0f}"
    )

    print(
        f"P90 predicted/S1    : "
        f"{best['p90_predicted']:.0f}"
    )

    print(
        f"Max predicted/S1    : "
        f"{best['max_predicted']:,}"
    )

    print(
        f"Perfect S1s         : "
        f"{best['perfect_s1']:,} / "
        f"{validation_s1_count:,}"
    )

    print(
        f"Exact set-match S1s : "
        f"{best['exact_set_match_s1']:,} / "
        f"{validation_s1_count:,}"
    )

    print()
    print(
        "Candidate ceiling pair recall: "
        f"{candidate_ceiling['candidate_pair_recall']:.4%}"
    )

    print(
        "Candidate ceiling fully-covered S1s: "
        f"{candidate_ceiling['s1_fully_covered']:,} / "
        f"{candidate_ceiling['s1_with_true']:,}"
    )

    return best


def save_report(
    results: list[dict],
    best: dict,
    candidate_ceiling: dict,
    validation_s1_count: int,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "XGBoost V2 S1-Level Evaluation\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            f"Validation S1 count: "
            f"{validation_s1_count:,}\n"
        )

        f.write(
            f"Candidate pair recall: "
            f"{candidate_ceiling['candidate_pair_recall']:.6%}\n"
        )

        f.write(
            f"Fully covered S1: "
            f"{candidate_ceiling['s1_fully_covered']:,} / "
            f"{candidate_ceiling['s1_with_true']:,}\n\n"
        )

        f.write(
            "Threshold Results\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        for result in results:

            f.write(
                f"Threshold {result['threshold']:.2f}\n"
            )

            f.write(
                f"  Macro Precision : "
                f"{result['macro_precision']:.6f}\n"
            )

            f.write(
                f"  Macro Recall    : "
                f"{result['macro_recall']:.6f}\n"
            )

            f.write(
                f"  Macro F0.5      : "
                f"{result['macro_f05']:.6f}\n"
            )

            f.write(
                f"  Micro Precision : "
                f"{result['micro_precision']:.6f}\n"
            )

            f.write(
                f"  Micro Recall    : "
                f"{result['micro_recall']:.6f}\n"
            )

            f.write(
                f"  Micro F0.5      : "
                f"{result['micro_f05']:.6f}\n"
            )

            f.write(
                f"  Mean Pred/S1    : "
                f"{result['mean_predicted']:.4f}\n"
            )

            f.write(
                f"  Exact S1        : "
                f"{result['exact_set_match_s1']:,}\n\n"
            )

        f.write(
            "\nBEST THRESHOLD\n"
        )

        f.write(
            "=" * 80 + "\n"
        )

        f.write(
            f"Threshold: "
            f"{best['threshold']:.2f}\n"
        )

        f.write(
            f"Macro F0.5: "
            f"{best['macro_f05']:.6f}\n"
        )

    with OUTPUT_THRESHOLD_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"{best['threshold']:.6f}\n"
        )

    print()
    print(
        f"Report saved: "
        f"{OUTPUT_REPORT_PATH}"
    )

    print(
        f"S1 threshold saved: "
        f"{OUTPUT_THRESHOLD_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("XGBOOST V2 — S1 LEVEL EVALUATION")
    print("=" * 70)

    print()
    print(f"Repository root : {ROOT}")
    print(f"ML database     : {ML_DB}")
    print(f"Model           : {MODEL_PATH}")
    print(f"Ground truth    : {GROUND_TRUTH_PATH}")

    # --------------------------------------------------------
    # 1. Load ML feature rows.
    # --------------------------------------------------------

    rows = load_ml_data()

    # --------------------------------------------------------
    # 2. Reproduce validation split.
    # --------------------------------------------------------

    (
        validation_s1_ids,
        validation_rows,
        X,
        y,
        candidate_s1,
        candidate_ids,
    ) = prepare_validation_data(
        rows
    )

    # --------------------------------------------------------
    # 3. Load ORIGINAL ground truth.
    # --------------------------------------------------------

    ground_truth = load_ground_truth(
        GROUND_TRUTH_PATH,
        validation_s1_ids,
    )

    # --------------------------------------------------------
    # 4. Sanity check.
    # --------------------------------------------------------

    missing_ground_truth = (
        validation_s1_ids
        - set(ground_truth.keys())
    )

    if missing_ground_truth:

        print()
        print(
            "WARNING:"
        )

        print(
            f"{len(missing_ground_truth):,} "
            "validation S1 IDs were not found "
            "in ground truth."
        )

    # --------------------------------------------------------
    # 5. Candidate coverage ceiling.
    # --------------------------------------------------------

    candidate_ceiling = (
        calculate_candidate_ceiling(
            validation_s1_ids,
            candidate_s1,
            candidate_ids,
            ground_truth,
        )
    )

    # --------------------------------------------------------
    # 6. XGBoost prediction.
    # --------------------------------------------------------

    probabilities = predict(
        MODEL_PATH,
        X,
    )

    # --------------------------------------------------------
    # 7. Threshold sweep.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATING THRESHOLDS")
    print("=" * 70)

    results = []

    for threshold in THRESHOLDS:

        print(
            f"Evaluating threshold "
            f"{threshold:.2f}..."
        )

        result = evaluate_threshold(
            threshold,
            validation_s1_ids,
            candidate_s1,
            candidate_ids,
            probabilities,
            ground_truth,
        )

        results.append(result)

    # --------------------------------------------------------
    # 8. Print results and choose best threshold.
    # --------------------------------------------------------

    best = print_results(
        results,
        candidate_ceiling,
        len(validation_s1_ids),
    )

    # --------------------------------------------------------
    # 9. Save results.
    # --------------------------------------------------------

    save_report(
        results,
        best,
        candidate_ceiling,
        len(validation_s1_ids),
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Recommended S1 threshold: "
        f"{best['threshold']:.2f}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This threshold is based on the "
        "100k-S1 validation experiment."
    )

    print(
        "It is NOT yet the final full-test "
        "submission threshold."
    )


if __name__ == "__main__":
    main()