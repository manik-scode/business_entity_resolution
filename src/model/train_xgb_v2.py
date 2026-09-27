from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import xgboost as xgb


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

ML_DB = ROOT / "results" / "ml" / "ml_train_v1.db"

MODEL_DIR = ROOT / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgb_entity_match_v2.json"

THRESHOLD_PATH = MODEL_DIR / "xgb_entity_match_v2_threshold.txt"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
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
# F0.5
# ============================================================

def f05(precision: float, recall: float) -> float:
    """
    F0.5 gives more importance to precision than recall.
    """

    beta2 = 0.5 ** 2

    denominator = (beta2 * precision) + recall

    if denominator == 0:
        return 0.0

    return (
        (1 + beta2)
        * precision
        * recall
        / denominator
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print()
    print("[ML] Opening SQLite database...")
    print(f"[ML] DB: {ML_DB}")

    conn = sqlite3.connect(ML_DB)

    columns = ", ".join(
        [
            "source1_entity_id",
            "label",
        ]
        + FEATURES
    )

    cursor = conn.execute(
        f"""
        SELECT {columns}
        FROM features
        ORDER BY source1_entity_id
        """
    )

    rows = cursor.fetchall()

    conn.close()

    print(f"[ML] Rows loaded: {len(rows):,}")

    return rows


# ============================================================
# BUILD NUMPY ARRAYS
# ============================================================

def build_arrays(rows):

    print()
    print("[ML] Building NumPy arrays...")

    s1_ids = np.array(
        [row[0] for row in rows],
        dtype=object,
    )

    y = np.array(
        [row[1] for row in rows],
        dtype=np.float32,
    )

    X = np.array(
        [row[2:] for row in rows],
        dtype=np.float32,
    )

    print()
    print("[ML] Feature matrix")
    print("[ML] ------------------------------")
    print(f"[ML] X shape : {X.shape}")
    print(f"[ML] y shape : {y.shape}")
    print(f"[ML] Positive: {int(y.sum()):,}")
    print(f"[ML] Negative: {int(len(y) - y.sum()):,}")

    return s1_ids, X, y


# ============================================================
# S1 LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

def make_s1_split(s1_ids):

    print()
    print("[ML] Creating S1-level split...")

    unique_s1 = np.unique(s1_ids)

    rng = np.random.default_rng(42)

    rng.shuffle(unique_s1)

    split_index = int(
        len(unique_s1) * 0.80
    )

    train_s1 = set(
        unique_s1[:split_index]
    )

    valid_s1 = set(
        unique_s1[split_index:]
    )

    train_mask = np.array(
        [
            s1 in train_s1
            for s1 in s1_ids
        ],
        dtype=bool,
    )

    valid_mask = np.array(
        [
            s1 in valid_s1
            for s1 in s1_ids
        ],
        dtype=bool,
    )

    print()
    print("[ML] S1 split")
    print("[ML] ------------------------------")
    print(f"[ML] Unique S1       : {len(unique_s1):,}")
    print(f"[ML] Train S1        : {len(train_s1):,}")
    print(f"[ML] Validation S1   : {len(valid_s1):,}")
    print(f"[ML] Train rows      : {int(train_mask.sum()):,}")
    print(f"[ML] Validation rows : {int(valid_mask.sum()):,}")

    return train_mask, valid_mask


# ============================================================
# TRAIN NATIVE XGBOOST
# ============================================================

def train_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
):

    print()
    print("[ML] Preparing XGBoost DMatrix...")
    print("[ML] ------------------------------")

    positives = int(y_train.sum())

    negatives = int(
        len(y_train) - y_train.sum()
    )

    scale_pos_weight = (
        negatives / positives
    )

    print(
        f"[ML] Train positives  : {positives:,}"
    )

    print(
        f"[ML] Train negatives  : {negatives:,}"
    )

    print(
        f"[ML] scale_pos_weight : "
        f"{scale_pos_weight:.4f}"
    )

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        feature_names=FEATURES,
    )

    dvalid = xgb.DMatrix(
        X_valid,
        label=y_valid,
        feature_names=FEATURES,
    )

    params = {
        "objective": "binary:logistic",

        "eval_metric": "logloss",

        "max_depth": 8,

        "learning_rate": 0.05,

        "subsample": 0.85,

        "colsample_bytree": 0.90,

        "min_child_weight": 5,

        "reg_lambda": 2.0,

        "scale_pos_weight": scale_pos_weight,

        "tree_method": "hist",

        # Use GPU if available.
        "device": "cuda",

        "seed": 42,
    }

    print()
    print("[ML] XGBoost configuration")
    print("[ML] ------------------------------")

    for key, value in params.items():
        print(
            f"[ML] {key:20s}: {value}"
        )

    print()
    print("[ML] Training XGBoost...")
    print("[ML] ------------------------------")

    evals_result = {}

    model = xgb.train(
        params=params,

        dtrain=dtrain,

        num_boost_round=700,

        evals=[
            (dtrain, "train"),
            (dvalid, "validation"),
        ],

        evals_result=evals_result,

        verbose_eval=50,

        early_stopping_rounds=50,
    )

    print()
    print("[ML] Training finished.")

    if model.best_iteration is not None:

        print(
            f"[ML] Best iteration : "
            f"{model.best_iteration}"
        )

    if model.best_score is not None:

        print(
            f"[ML] Best score     : "
            f"{model.best_score:.6f}"
        )

    return model, dvalid


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def calculate_metrics(
    probabilities,
    y_true,
    threshold,
):

    predictions = (
        probabilities >= threshold
    )

    tp = int(
        np.sum(
            (predictions == 1)
            & (y_true == 1)
        )
    )

    fp = int(
        np.sum(
            (predictions == 1)
            & (y_true == 0)
        )
    )

    fn = int(
        np.sum(
            (predictions == 0)
            & (y_true == 1)
        )
    )

    tn = int(
        np.sum(
            (predictions == 0)
            & (y_true == 0)
        )
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f05_score = f05(
        precision,
        recall,
    )

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f05": f05_score,
    }


def evaluate_thresholds(
    model,
    dvalid,
    y_valid,
):

    print()
    print("[ML] Generating validation probabilities...")
    print("[ML] ------------------------------")

    probabilities = model.predict(
        dvalid
    )

    print(
        f"[ML] Predictions: "
        f"{len(probabilities):,}"
    )

    print()
    print("[ML] Probability statistics")
    print("[ML] ------------------------------")

    print(
        f"[ML] Minimum : {probabilities.min():.6f}"
    )

    print(
        f"[ML] Maximum : {probabilities.max():.6f}"
    )

    print(
        f"[ML] Mean    : {probabilities.mean():.6f}"
    )

    # --------------------------------------------------------
    # Fine threshold sweep
    # --------------------------------------------------------

    thresholds = np.arange(
        0.01,
        1.0001,
        0.01,
    )

    results = []

    for threshold in thresholds:

        result = calculate_metrics(
            probabilities,
            y_valid,
            float(threshold),
        )

        results.append(result)

    best = max(
        results,
        key=lambda x: x["f05"],
    )

    # --------------------------------------------------------
    # Display selected thresholds
    # --------------------------------------------------------

    print()
    print("[ML] Threshold sweep")
    print("[ML] ------------------------------")

    print(
        f"{'Threshold':>10} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F0.5':>10} "
        f"{'TP':>10} "
        f"{'FP':>10}"
    )

    print("-" * 72)

    display_thresholds = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.85,
        0.90,
        0.92,
        0.94,
        0.95,
        0.96,
        0.97,
        0.98,
        0.99,
    ]

    for threshold in display_thresholds:

        result = calculate_metrics(
            probabilities,
            y_valid,
            threshold,
        )

        print(
            f"{threshold:10.2f} "
            f"{result['precision']:10.4f} "
            f"{result['recall']:10.4f} "
            f"{result['f05']:10.4f} "
            f"{result['tp']:10,} "
            f"{result['fp']:10,}"
        )

    # --------------------------------------------------------
    # Best result
    # --------------------------------------------------------

    print()
    print("[ML] BEST VALIDATION RESULT")
    print("[ML] ==============================")

    print(
        f"[ML] Threshold : "
        f"{best['threshold']:.2f}"
    )

    print(
        f"[ML] Precision : "
        f"{best['precision']:.6f}"
    )

    print(
        f"[ML] Recall    : "
        f"{best['recall']:.6f}"
    )

    print(
        f"[ML] F0.5      : "
        f"{best['f05']:.6f}"
    )

    print(
        f"[ML] TP        : "
        f"{best['tp']:,}"
    )

    print(
        f"[ML] FP        : "
        f"{best['fp']:,}"
    )

    print(
        f"[ML] FN        : "
        f"{best['fn']:,}"
    )

    print(
        f"[ML] TN        : "
        f"{best['tn']:,}"
    )

    return best


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def print_feature_importance(model):

    print()
    print("[ML] Feature importance")
    print("[ML] ------------------------------")

    importance = model.get_score(
        importance_type="gain"
    )

    ranked = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    if not ranked:
        print("[ML] No feature importance available.")
        return

    print(
        f"{'Feature':35s} {'Gain':>15s}"
    )

    print("-" * 52)

    for feature, gain in ranked:

        print(
            f"{feature:35s} "
            f"{gain:15.4f}"
        )


# ============================================================
# SAVE MODEL + THRESHOLD
# ============================================================

def save_artifacts(
    model,
    best,
):

    print()
    print("[ML] Saving artifacts...")
    print("[ML] ------------------------------")

    model.save_model(
        MODEL_PATH
    )

    THRESHOLD_PATH.write_text(
        str(best["threshold"]),
        encoding="utf-8",
    )

    print(
        f"[ML] Model     : {MODEL_PATH}"
    )

    print(
        f"[ML] Threshold : {THRESHOLD_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("========================================")
    print("XGBOOST ENTITY MATCH MODEL V2")
    print("NATIVE XGBOOST API")
    print("========================================")

    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    if not ML_DB.exists():

        raise FileNotFoundError(
            f"ML database not found:\n{ML_DB}"
        )

    # --------------------------------------------------------
    # Verify XGBoost
    # --------------------------------------------------------

    print()
    print(
        f"[ML] XGBoost version: "
        f"{xgb.__version__}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    rows = load_data()

    if not rows:

        raise RuntimeError(
            "No feature rows found."
        )

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    s1_ids, X, y = build_arrays(
        rows
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_mask, valid_mask = (
        make_s1_split(s1_ids)
    )

    X_train = X[train_mask]

    y_train = y[train_mask]

    X_valid = X[valid_mask]

    y_valid = y[valid_mask]

    # --------------------------------------------------------
    # Free unused arrays
    # --------------------------------------------------------

    del rows
    del s1_ids
    del X
    del y

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model, dvalid = train_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    print_feature_importance(
        model
    )

    # --------------------------------------------------------
    # Evaluate thresholds
    # --------------------------------------------------------

    best = evaluate_thresholds(
        model,
        dvalid,
        y_valid,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_artifacts(
        model,
        best,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("========================================")
    print("TRAINING COMPLETE")
    print("========================================")

    print(
        f"[ML] Validation F0.5 : "
        f"{best['f05']:.6f}"
    )

    print(
        f"[ML] Best threshold   : "
        f"{best['threshold']:.2f}"
    )

    print(
        f"[ML] Precision         : "
        f"{best['precision']:.6f}"
    )

    print(
        f"[ML] Recall            : "
        f"{best['recall']:.6f}"
    )

    print()
    print("[ML] Model:")
    print(MODEL_PATH)

    print()
    print("[ML] Threshold:")
    print(THRESHOLD_PATH)

    print()
    print("[ML] IMPORTANT:")
    print(
        "[ML] This is validation performance only."
    )
    print(
        "[ML] Do not use this threshold on the "
        "full test set until validation is reviewed."
    )


if __name__ == "__main__":
    main()