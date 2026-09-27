from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[2]

ML_DB = ROOT / "results" / "ml" / "ml_train_v1.db"
MODEL_DIR = ROOT / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgb_entity_match_v1.json"


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


def f05(precision: float, recall: float) -> float:
    """F0.5: precision gets twice the weight of recall."""
    beta2 = 0.5 ** 2

    denominator = beta2 * precision + recall

    if denominator == 0:
        return 0.0

    return (1 + beta2) * precision * recall / denominator


def load_data():
    print("[ML] Opening SQLite database...")

    conn = sqlite3.connect(ML_DB)

    columns = ", ".join(
        ["source1_entity_id", "label"] + FEATURES
    )

    rows = conn.execute(
        f"""
        SELECT {columns}
        FROM features
        ORDER BY source1_entity_id
        """
    )

    data = list(rows)

    conn.close()

    print(f"[ML] Rows loaded: {len(data):,}")

    return data


def build_arrays(rows):
    s1_ids = np.array(
        [row[0] for row in rows],
        dtype=object,
    )

    y = np.array(
        [row[1] for row in rows],
        dtype=np.int8,
    )

    X = np.array(
        [row[2:] for row in rows],
        dtype=np.float32,
    )

    return s1_ids, X, y


def make_s1_split(s1_ids):
    """
    Split by S1 entity, NOT by pair.

    This prevents the same S1 entity from appearing
    in both training and validation.
    """

    unique_s1 = np.unique(s1_ids)

    rng = np.random.default_rng(42)

    rng.shuffle(unique_s1)

    split_index = int(len(unique_s1) * 0.80)

    train_s1 = set(unique_s1[:split_index])
    valid_s1 = set(unique_s1[split_index:])

    train_mask = np.array(
        [x in train_s1 for x in s1_ids],
        dtype=bool,
    )

    valid_mask = np.array(
        [x in valid_s1 for x in s1_ids],
        dtype=bool,
    )

    print()
    print("[ML] S1 split")
    print("[ML] ------------------------------")
    print(f"[ML] Unique S1       : {len(unique_s1):,}")
    print(f"[ML] Train S1        : {len(train_s1):,}")
    print(f"[ML] Validation S1   : {len(valid_s1):,}")
    print(f"[ML] Train rows      : {train_mask.sum():,}")
    print(f"[ML] Validation rows : {valid_mask.sum():,}")

    return train_mask, valid_mask


def train_model(X_train, y_train, X_valid, y_valid):

    print()
    print("[ML] Training XGBoost...")
    print("[ML] ------------------------------")

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)

    scale_pos_weight = negatives / positives

    print(f"[ML] Train positives     : {positives:,}")
    print(f"[ML] Train negatives     : {negatives:,}")
    print(f"[ML] scale_pos_weight    : {scale_pos_weight:.3f}")

    model = xgb.XGBClassifier(
        n_estimators=700,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.90,
        min_child_weight=5,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=6,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=50,
    )

    return model


def evaluate_thresholds(model, X_valid, y_valid):

    print()
    print("[ML] Threshold sweep")
    print("[ML] ------------------------------")

    probabilities = model.predict_proba(X_valid)[:, 1]

    best = None

    thresholds = np.arange(
        0.05,
        1.00,
        0.01,
    )

    for threshold in thresholds:

        predictions = probabilities >= threshold

        tp = int(np.sum((predictions == 1) & (y_valid == 1)))
        fp = int(np.sum((predictions == 1) & (y_valid == 0)))
        fn = int(np.sum((predictions == 0) & (y_valid == 1)))

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        score = f05(
            precision,
            recall,
        )

        result = {
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f05": score,
        }

        if best is None or score > best["f05"]:
            best = result

    print()
    print(
        f"{'Threshold':>10} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F0.5':>10} "
        f"{'TP':>10} "
        f"{'FP':>10}"
    )

    print("-" * 70)

    # Show selected thresholds for context.
    for threshold in [
        0.50,
        0.70,
        0.80,
        0.90,
        0.95,
        0.97,
        0.98,
        0.99,
    ]:

        probabilities = model.predict_proba(X_valid)[:, 1]

        predictions = probabilities >= threshold

        tp = int(np.sum((predictions == 1) & (y_valid == 1)))
        fp = int(np.sum((predictions == 1) & (y_valid == 0)))
        fn = int(np.sum((predictions == 0) & (y_valid == 1)))

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        score = f05(
            precision,
            recall,
        )

        print(
            f"{threshold:10.2f} "
            f"{precision:10.4f} "
            f"{recall:10.4f} "
            f"{score:10.4f} "
            f"{tp:10,} "
            f"{fp:10,}"
        )

    print()
    print("[ML] BEST THRESHOLD")
    print("[ML] ------------------------------")
    print(f"[ML] Threshold : {best['threshold']:.2f}")
    print(f"[ML] Precision : {best['precision']:.4f}")
    print(f"[ML] Recall    : {best['recall']:.4f}")
    print(f"[ML] F0.5      : {best['f05']:.4f}")
    print(f"[ML] TP        : {best['tp']:,}")
    print(f"[ML] FP        : {best['fp']:,}")
    print(f"[ML] FN        : {best['fn']:,}")

    return best


def main():

    print()
    print("========================================")
    print("XGBOOST ENTITY MATCH MODEL V1")
    print("========================================")

    if not ML_DB.exists():
        raise FileNotFoundError(
            f"ML database not found: {ML_DB}"
        )

    rows = load_data()

    s1_ids, X, y = build_arrays(rows)

    print()
    print("[ML] Feature matrix")
    print("[ML] ------------------------------")
    print(f"[ML] X shape : {X.shape}")
    print(f"[ML] y shape : {y.shape}")
    print(f"[ML] Positive: {int(y.sum()):,}")
    print(f"[ML] Negative: {int(len(y) - y.sum()):,}")

    train_mask, valid_mask = make_s1_split(s1_ids)

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    model = train_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
    )

    best = evaluate_thresholds(
        model,
        X_valid,
        y_valid,
    )

    model.save_model(MODEL_PATH)

    print()
    print("[ML] Model saved:")
    print(f"[ML] {MODEL_PATH}")

    print()
    print("========================================")
    print("TRAINING COMPLETE")
    print("========================================")
    print(
        f"Best validation F0.5: {best['f05']:.4f}"
    )


if __name__ == "__main__":
    main()