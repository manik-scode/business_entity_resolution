import argparse
import csv
import sqlite3
import time
from pathlib import Path


def load_ground_truth(path):
    gt = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            s1_id = row["source1_entity_id"]
            raw = (row["matched_entity_ids"] or "").strip()

            if raw:
                gt[s1_id] = set(raw.split(","))
            else:
                gt[s1_id] = set()

    return gt


def load_candidates(path):
    candidates = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            s1_id = row["source1_entity_id"]
            candidate_id = row["candidate_entity_id"]

            candidates.setdefault(s1_id, set()).add(candidate_id)

    return candidates


def load_s1_order(source1_path, limit):
    s1_ids = []
    if source1_path and Path(source1_path).exists():
        with open(source1_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            col = "entity_id" if "entity_id" in reader.fieldnames else "source1_entity_id"
            for row in reader:
                s1_ids.append(row[col])
                if limit and len(s1_ids) >= limit:
                    break
    return s1_ids


def main(args):
    start = time.time()

    print("Loading ground truth...")
    gt = load_ground_truth(args.ground_truth)

    print("Loading candidates...")
    candidates = load_candidates(args.candidates)

    # Determine the evaluated entities in the exact order they were processed
    s1_order = []
    if args.matching and Path(args.matching).exists():
        s1_order = load_s1_order(args.matching, args.limit)
    elif args.source1 and Path(args.source1).exists():
        s1_order = load_s1_order(args.source1, args.limit)

    evaluated = 0
    true_matches = 0
    retrieved_true = 0

    perfect_entities = 0
    zero_candidate_entities = 0

    total_candidates = 0

    eval_items = []
    if s1_order:
        eval_items = [(s1_id, gt.get(s1_id, set())) for s1_id in s1_order]
    else:
        eval_items = list(gt.items())

    for s1_id, true_ids in eval_items:

        if args.limit and evaluated >= args.limit:
            break

        pred_ids = candidates.get(s1_id, set())

        total_candidates += len(pred_ids)

        retrieved = true_ids & pred_ids

        true_matches += len(true_ids)
        retrieved_true += len(retrieved)

        if not pred_ids:
            zero_candidate_entities += 1

        if true_ids and true_ids.issubset(pred_ids):
            perfect_entities += 1

        evaluated += 1

    recall = (
        retrieved_true / true_matches
        if true_matches
        else 0.0
    )

    avg_candidates = (
        total_candidates / evaluated
        if evaluated
        else 0.0
    )

    print()
    print("=" * 70)
    print("FAST V4 TRAIN CANDIDATE EVALUATION")
    print("=" * 70)

    print(f"S1 evaluated           : {evaluated:,}")
    print(f"True matches           : {true_matches:,}")
    print(f"Retrieved true matches : {retrieved_true:,}")

    print(
        f"Candidate recall       : "
        f"{recall * 100:.4f}%"
    )

    print(
        f"Average candidates    : "
        f"{avg_candidates:.2f}"
    )

    print(
        f"Perfect entity recall : "
        f"{perfect_entities:,} "
        f"({perfect_entities / evaluated * 100:.2f}%)"
    )

    print(
        f"Zero-candidate S1     : "
        f"{zero_candidate_entities:,} "
        f"({zero_candidate_entities / evaluated * 100:.2f}%)"
    )

    print(
        f"Evaluation time       : "
        f"{(time.time() - start):.2f} sec"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ground-truth",
        required=True
    )

    parser.add_argument(
        "--candidates",
        required=True
    )

    parser.add_argument(
        "--source1",
        default="student_resource/dataset/train/train_source1.tsv"
    )

    parser.add_argument(
        "--matching",
        default="results/matching_results_fast_train_100k.tsv"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0
    )

    args = parser.parse_args()

    main(args)
