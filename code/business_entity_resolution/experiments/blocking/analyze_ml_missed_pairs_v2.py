import csv
import sqlite3
import time
from pathlib import Path
from collections import Counter

from build_index_v2 import (
    extract_name_keys,
    normalize_text,
    extract_address_number,
    name_prefix_signature,
    name_suffix_signature,
    address_prefix_signature,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(".")

MISSED_FILE = ROOT / "results" / "blocking" / "ml_missed_pairs_v1.tsv"

S1_FILE = (
    ROOT
    / "student_resource"
    / "dataset"
    / "train"
    / "train_source1.tsv"
)

DB_FILE = (
    ROOT
    / "results"
    / "blocking"
    / "blocking_train_v2.db"
)

OUTPUT_FILE = (
    ROOT
    / "results"
    / "blocking"
    / "ml_missed_pairs_analysis_v2.tsv"
)


# ============================================================
# KEY GENERATION
# EXACTLY REUSES build_index_v2.py LOGIC
# ============================================================

def build_keys(name, address):
    (
        name_key,
        sorted_name_key,
        name_token_key,
        first_name_token,
    ) = extract_name_keys(name)

    address_key = normalize_text(address)

    address_number = extract_address_number(address_key)

    prefix_sig = name_prefix_signature(name_key)
    suffix_sig = name_suffix_signature(name_key)
    address_token_sig = address_prefix_signature(address_key)

    return {
        "name_key": name_key,
        "sorted_name_key": sorted_name_key,
        "name_token_key": name_token_key,
        "first_name_token": first_name_token,
        "address_key": address_key,
        "address_number": address_number,
        "name_prefix_sig": prefix_sig,
        "name_suffix_sig": suffix_sig,
        "address_token_sig": address_token_sig,
    }


# ============================================================
# LOAD MISSED PAIRS
# ============================================================

def load_missed_pairs():
    pairs = []

    required_s1 = set()
    required_candidates = set()

    with open(
        MISSED_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            s1 = row["source1_entity_id"]
            cid = row["true_candidate_entity_id"]

            pairs.append((s1, cid))

            required_s1.add(s1)
            required_candidates.add(cid)

    return pairs, required_s1, required_candidates


# ============================================================
# LOAD S1 RECORDS
# ONLY REQUIRED S1 IDS ARE KEPT
# ============================================================

def load_required_s1(required_ids):

    print()
    print("Scanning train_source1.tsv...")
    print(f"Required S1 records: {len(required_ids):,}")

    records = {}

    start = time.time()

    with open(
        S1_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        scanned = 0

        for row in reader:

            scanned += 1

            entity_id = row["entity_id"]

            if entity_id not in required_ids:
                continue

            name = row["business_name"] or ""
            address = row["business_address"] or ""
            country = row["country"] or ""

            keys = build_keys(name, address)

            records[entity_id] = {
                "entity_id": entity_id,
                "source": "S1",
                "country": country,
                "business_name": name,
                "business_address": address,
                **keys,
            }

            if scanned % 500_000 == 0:
                print(
                    f"  scanned {scanned:,} rows | "
                    f"found {len(records):,}"
                )

    print(
        f"S1 records loaded: {len(records):,}"
    )

    print(
        f"S1 scan time: {time.time() - start:.1f}s"
    )

    missing = required_ids - records.keys()

    if missing:
        print(
            f"WARNING: {len(missing):,} required S1 records "
            f"were not found."
        )

    return records


# ============================================================
# LOAD REQUIRED S2/S3 RECORDS FROM EXISTING V2 DB
# ============================================================

def load_required_candidates(conn, required_ids):

    print()
    print("Loading required S2/S3 records from blocking_train_v2.db...")
    print(f"Required candidate records: {len(required_ids):,}")

    records = {}

    cursor = conn.cursor()

    ids = list(required_ids)

    batch_size = 500

    for start in range(0, len(ids), batch_size):

        batch = ids[start:start + batch_size]

        placeholders = ",".join("?" for _ in batch)

        query = f"""
            SELECT
                entity_id,
                source,
                country,
                business_name,
                business_address,
                name_key,
                sorted_name_key,
                name_token_key,
                first_name_token,
                address_number,
                address_key,
                name_prefix_sig,
                name_suffix_sig,
                address_token_sig
            FROM records
            WHERE entity_id IN ({placeholders})
        """

        for row in cursor.execute(query, batch):

            (
                entity_id,
                source,
                country,
                business_name,
                business_address,
                name_key,
                sorted_name_key,
                name_token_key,
                first_name_token,
                address_number,
                address_key,
                name_prefix_sig,
                name_suffix_sig,
                address_token_sig,
            ) = row

            records[entity_id] = {
                "entity_id": entity_id,
                "source": source,
                "country": country,
                "business_name": business_name or "",
                "business_address": business_address or "",
                "name_key": name_key or "",
                "sorted_name_key": sorted_name_key or "",
                "name_token_key": name_token_key or "",
                "first_name_token": first_name_token or "",
                "address_key": address_key or "",
                "address_number": address_number or "",
                "name_prefix_sig": name_prefix_sig or "",
                "name_suffix_sig": name_suffix_sig or "",
                "address_token_sig": address_token_sig or "",
            }

    print(
        f"S2/S3 records loaded: {len(records):,}"
    )

    missing = required_ids - records.keys()

    if missing:
        print(
            f"WARNING: {len(missing):,} required S2/S3 records "
            f"were not found."
        )

    return records


# ============================================================
# RULE EVALUATION
# ============================================================

def evaluate_pair(s1, candidate):

    same_country = (
        s1["country"] == candidate["country"]
    )

    if not same_country:
        return {
            "name_exact": 0,
            "sorted_name": 0,
            "token_name": 0,
            "first_name": 0,
            "address_exact": 0,
            "address_number": 0,
            "address_token": 0,
            "prefix": 0,
            "suffix": 0,
            "r4": 0,
            "r5": 0,
            "prefix_number": 0,
            "suffix_number": 0,
        }

    return {
        # Existing V2 rules
        "name_exact": int(
            bool(s1["name_key"])
            and s1["name_key"] == candidate["name_key"]
        ),

        "sorted_name": int(
            bool(s1["sorted_name_key"])
            and s1["sorted_name_key"]
            == candidate["sorted_name_key"]
        ),

        "token_name": int(
            bool(s1["name_token_key"])
            and s1["name_token_key"]
            == candidate["name_token_key"]
        ),

        "first_name": int(
            bool(s1["first_name_token"])
            and s1["first_name_token"]
            == candidate["first_name_token"]
        ),

        "address_exact": int(
            bool(s1["address_key"])
            and s1["address_key"]
            == candidate["address_key"]
        ),

        "address_number": int(
            bool(s1["address_number"])
            and s1["address_number"]
            == candidate["address_number"]
        ),

        "address_token": int(
            bool(s1["address_token_sig"])
            and s1["address_token_sig"]
            == candidate["address_token_sig"]
        ),

        "prefix": int(
            bool(s1["name_prefix_sig"])
            and s1["name_prefix_sig"]
            == candidate["name_prefix_sig"]
        ),

        "suffix": int(
            bool(s1["name_suffix_sig"])
            and s1["name_suffix_sig"]
            == candidate["name_suffix_sig"]
        ),

        # V13
        "r4": int(
            bool(s1["address_number"])
            and bool(s1["address_token_sig"])
            and s1["address_number"]
            == candidate["address_number"]
            and s1["address_token_sig"]
            == candidate["address_token_sig"]
        ),

        "r5": int(
            bool(s1["first_name_token"])
            and bool(s1["address_token_sig"])
            and s1["first_name_token"]
            == candidate["first_name_token"]
            and s1["address_token_sig"]
            == candidate["address_token_sig"]
        ),

        # Additional combinations
        "prefix_number": int(
            bool(s1["name_prefix_sig"])
            and bool(s1["address_number"])
            and s1["name_prefix_sig"]
            == candidate["name_prefix_sig"]
            and s1["address_number"]
            == candidate["address_number"]
        ),

        "suffix_number": int(
            bool(s1["name_suffix_sig"])
            and bool(s1["address_number"])
            and s1["name_suffix_sig"]
            == candidate["name_suffix_sig"]
            and s1["address_number"]
            == candidate["address_number"]
        ),
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    start = time.time()

    print("=" * 75)
    print("ML MISSED TRUE-PAIR FORENSIC ANALYSIS V2")
    print("=" * 75)

    # --------------------------------------------------------
    # Load missed pairs
    # --------------------------------------------------------

    pairs, required_s1, required_candidates = (
        load_missed_pairs()
    )

    print()
    print(f"Missed true pairs       : {len(pairs):,}")
    print(f"Unique S1 IDs           : {len(required_s1):,}")
    print(
        f"Unique candidate IDs    : "
        f"{len(required_candidates):,}"
    )

    # --------------------------------------------------------
    # Load records
    # --------------------------------------------------------

    s1_records = load_required_s1(
        required_s1
    )

    conn = sqlite3.connect(
        DB_FILE
    )

    conn.execute(
        "PRAGMA query_only=ON"
    )

    candidate_records = load_required_candidates(
        conn,
        required_candidates,
    )

    conn.close()

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    rule_names = [
        "name_exact",
        "sorted_name",
        "token_name",
        "first_name",
        "address_exact",
        "address_number",
        "address_token",
        "prefix",
        "suffix",
        "r4",
        "r5",
        "prefix_number",
        "suffix_number",
    ]

    counts = Counter()

    analyzable = 0
    no_signal = 0

    output_rows = []

    for index, (s1_id, candidate_id) in enumerate(pairs, 1):

        s1 = s1_records.get(s1_id)
        candidate = candidate_records.get(candidate_id)

        if s1 is None or candidate is None:
            continue

        analyzable += 1

        signals = evaluate_pair(
            s1,
            candidate,
        )

        matched_rules = [
            rule
            for rule in rule_names
            if signals[rule]
        ]

        if not matched_rules:
            no_signal += 1

        for rule in matched_rules:
            counts[rule] += 1

        output_rows.append(
            [
                s1_id,
                candidate_id,
                s1["country"],
                candidate["source"],
                "|".join(matched_rules),
                s1["business_name"],
                candidate["business_name"],
                s1["business_address"],
                candidate["business_address"],
            ]
        )

        if index % 2_000 == 0:
            print(
                f"Analyzed {index:,}/{len(pairs):,} "
                f"| analyzable={analyzable:,}"
            )

    # --------------------------------------------------------
    # Save detailed output
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
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
                "candidate_entity_id",
                "country",
                "candidate_source",
                "matched_blocking_rules",
                "s1_business_name",
                "candidate_business_name",
                "s1_business_address",
                "candidate_business_address",
            ]
        )

        writer.writerows(
            output_rows
        )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("RESULTS")
    print("=" * 75)

    print(
        f"Missed true pairs       : {len(pairs):,}"
    )

    print(
        f"Analyzable pairs        : {analyzable:,}"
    )

    print(
        f"Unavailable records     : "
        f"{len(pairs) - analyzable:,}"
    )

    if analyzable == 0:
        print()
        print(
            "ERROR: No analyzable pairs."
        )
        return

    print()

    for rule in rule_names:

        recovered = counts[rule]

        print(
            f"{rule:<20} "
            f"{recovered:>8,} "
            f"({recovered / analyzable:.2%})"
        )

    # --------------------------------------------------------
    # Union of existing exact rules
    # --------------------------------------------------------

    union_existing = 0
    union_r4_r5 = 0
    union_all = 0

    for row in output_rows:

        rules = set(
            row[4].split("|")
        ) if row[4] else set()

        existing = {
            "name_exact",
            "sorted_name",
            "token_name",
            "first_name",
            "address_exact",
        }

        if rules & existing:
            union_existing += 1

        if rules & {"r4", "r5"}:
            union_r4_r5 += 1

        if rules:
            union_all += 1

    print()
    print("-" * 75)

    print(
        f"Existing exact-name/address union : "
        f"{union_existing:,} "
        f"({union_existing / analyzable:.2%})"
    )

    print(
        f"R4 + R5 recovery                  : "
        f"{union_r4_r5:,} "
        f"({union_r4_r5 / analyzable:.2%})"
    )

    print(
        f"Any tested exact signal            : "
        f"{union_all:,} "
        f"({union_all / analyzable:.2%})"
    )

    no_signal = analyzable - union_all

    print(
        f"No tested exact signal             : "
        f"{no_signal:,} "
        f"({no_signal / analyzable:.2%})"
    )

    print()
    print(
        f"Detailed output: {OUTPUT_FILE}"
    )

    print(
        f"Total runtime: {time.time() - start:.1f}s"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()