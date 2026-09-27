import argparse
import csv
import re
import sqlite3
import time
import unicodedata
from collections import Counter


COUNTRY_MAP = {
    "INDIA": "India",
    "US": "US",
    "FRANCE": "France",
}


IGNORED_NAME_TERMS = {
    "corp", "co", "inc", "ltd", "pvt", "pub",
    "llc", "llp", "plc", "sa", "sarl", "gmbh",
    "ag", "limited", "company", "corporation"
}


GENERIC_ADDRESS_TOKENS = {
    "road", "rd", "street", "st", "lane", "ln",
    "avenue", "ave", "boulevard", "blvd",
    "highway", "hwy", "sector", "sec",
    "block", "district", "dist", "city",
    "town", "village", "near", "opposite",
    "opp", "india", "usa", "us", "france"
}


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()

    replacements = {
        "corporation": "corp",
        "company": "co",
        "incorporated": "inc",
        "limited": "ltd",
        "private": "pvt",
        "public": "pub",
        "societe anonyme": "sa",
        "societe": "soc",
        "etablissement": "etb",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def name_features(name):
    normalized = normalize_text(name)
    tokens = [
        x for x in normalized.split()
        if x and x not in IGNORED_NAME_TERMS
    ]

    if not tokens:
        tokens = normalized.split()

    compact = "".join(tokens)

    sorted_tokens = sorted(set(tokens))

    name_key = " ".join(tokens)
    sorted_name_key = " ".join(sorted_tokens)
    token_key = "|".join(sorted_tokens)

    first_token = tokens[0] if tokens else ""

    prefix_sig = (
        f"{compact[:4]}:{(len(compact) // 5) * 5}"
        if compact else ""
    )

    suffix_sig = (
        f"{compact[-4:]}:{(len(compact) // 5) * 5}"
        if compact else ""
    )

    return {
        "name_key": name_key,
        "sorted_name_key": sorted_name_key,
        "token_key": token_key,
        "first_token": first_token,
        "prefix_sig": prefix_sig,
        "suffix_sig": suffix_sig,
    }


def address_features(address):
    normalized = normalize_text(address)

    tokens = [
        x for x in normalized.split()
        if x not in GENERIC_ADDRESS_TOKENS
    ]

    numbers = re.findall(r"\b\d+[a-z]?\b", normalized)

    address_number = numbers[0] if numbers else ""

    if tokens:
        informative = max(tokens, key=len)
    else:
        informative = ""

    address_token_sig = informative[:5]

    return {
        "address_key": normalized,
        "address_number": address_number,
        "address_token_sig": address_token_sig,
    }


def get_features(row):
    nf = name_features(row.get("business_name", ""))
    af = address_features(row.get("business_address", ""))

    country_raw = (row.get("country") or "").strip()
    country = COUNTRY_MAP.get(country_raw.upper(), country_raw)

    return {
        "country": country,
        **nf,
        **af,
    }


def query_ids(conn, sql, params):
    cur = conn.execute(sql, params)
    return {row[0] for row in cur.fetchall()}


def get_candidates(conn, f):
    country = f["country"]

    candidates = set()
    rule_hits = Counter()

    # ---------------------------------------------------------
    # R1: Exact normalized name
    # ---------------------------------------------------------
    ids = query_ids(
        conn,
        """
        SELECT entity_id
        FROM records
        WHERE country = ?
          AND name_key = ?
        """,
        (country, f["name_key"])
    )

    candidates.update(ids)
    rule_hits["R1_exact_name"] = len(ids)

    # ---------------------------------------------------------
    # R2: Word-order normalized name
    # ---------------------------------------------------------
    ids = query_ids(
        conn,
        """
        SELECT entity_id
        FROM records
        WHERE country = ?
          AND sorted_name_key = ?
        """,
        (country, f["sorted_name_key"])
    )

    candidates.update(ids)
    rule_hits["R2_sorted_name"] = len(ids)

    # ---------------------------------------------------------
    # R3: Name token + address number
    # ---------------------------------------------------------
    if f["token_key"] and f["address_number"]:
        ids = query_ids(
            conn,
            """
            SELECT entity_id
            FROM records
            WHERE country = ?
              AND name_token_key = ?
              AND address_number = ?
            """,
            (
                country,
                f["token_key"],
                f["address_number"],
            )
        )

        candidates.update(ids)
        rule_hits["R3_token_address_number"] = len(ids)

    # ---------------------------------------------------------
    # R4: First name token + address number
    # ---------------------------------------------------------
    if f["first_token"] and f["address_number"]:
        ids = query_ids(
            conn,
            """
            SELECT entity_id
            FROM records
            WHERE country = ?
              AND first_name_token = ?
              AND address_number = ?
            """,
            (
                country,
                f["first_token"],
                f["address_number"],
            )
        )

        candidates.update(ids)
        rule_hits["R4_first_token_address_number"] = len(ids)

    # ---------------------------------------------------------
    # R5: Exact normalized address
    # ---------------------------------------------------------
    if f["address_key"]:
        ids = query_ids(
            conn,
            """
            SELECT entity_id
            FROM records
            WHERE country = ?
              AND address_key = ?
            """,
            (country, f["address_key"])
        )

        candidates.update(ids)
        rule_hits["R5_exact_address"] = len(ids)

    # ---------------------------------------------------------
    # R6: Prefix + address number
    # ---------------------------------------------------------
    if f["prefix_sig"] and f["address_number"]:
        ids = query_ids(
            conn,
            """
            SELECT entity_id
            FROM records
            WHERE country = ?
              AND name_prefix_sig = ?
              AND address_number = ?
            """,
            (
                country,
                f["prefix_sig"],
                f["address_number"],
            )
        )

        candidates.update(ids)
        rule_hits["R6_prefix_address_number"] = len(ids)

    # ---------------------------------------------------------
    # R7: Suffix + address number
    # ---------------------------------------------------------
    if f["suffix_sig"] and f["address_number"]:
        ids = query_ids(
            conn,
            """
            SELECT entity_id
            FROM records
            WHERE country = ?
              AND name_suffix_sig = ?
              AND address_number = ?
            """,
            (
                country,
                f["suffix_sig"],
                f["address_number"],
            )
        )

        candidates.update(ids)
        rule_hits["R7_suffix_address_number"] = len(ids)

    # ---------------------------------------------------------
    # R8: Prefix + address token
    # ---------------------------------------------------------
    if f["prefix_sig"] and f["address_token_sig"]:
        ids = query_ids(
            conn,
            """
            SELECT entity_id
            FROM records
            WHERE country = ?
              AND name_prefix_sig = ?
              AND address_token_sig = ?
            """,
            (
                country,
                f["prefix_sig"],
                f["address_token_sig"],
            )
        )

        candidates.update(ids)
        rule_hits["R8_prefix_address_token"] = len(ids)

    # ---------------------------------------------------------
    # R9: Suffix + address token
    # ---------------------------------------------------------
    if f["suffix_sig"] and f["address_token_sig"]:
        ids = query_ids(
            conn,
            """
            SELECT entity_id
            FROM records
            WHERE country = ?
              AND name_suffix_sig = ?
              AND address_token_sig = ?
            """,
            (
                country,
                f["suffix_sig"],
                f["address_token_sig"],
            )
        )

        candidates.update(ids)
        rule_hits["R9_suffix_address_token"] = len(ids)

    # ---------------------------------------------------------
    # R10: First name token + address token
    # ---------------------------------------------------------
    if f["first_token"] and f["address_token_sig"]:
        ids = query_ids(
            conn,
            """
            SELECT entity_id
            FROM records
            WHERE country = ?
              AND first_name_token = ?
              AND address_token_sig = ?
            """,
            (
                country,
                f["first_token"],
                f["address_token_sig"],
            )
        )

        candidates.update(ids)
        rule_hits["R10_first_token_address_token"] = len(ids)

    return candidates, rule_hits


def load_ground_truth(path):
    truth = {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            s1_id = row["source1_entity_id"]
            raw = (row["matched_entity_ids"] or "").strip()

            if raw:
                matches = set(
                    x.strip()
                    for x in raw.split(",")
                    if x.strip()
                )
            else:
                matches = set()

            truth[s1_id] = matches

    return truth


def evaluate(args):
    truth = load_ground_truth(args.ground_truth)

    conn = sqlite3.connect(args.db)

    total_true = 0
    total_retrieved_true = 0
    total_candidates = 0

    candidate_counts = []
    rule_total = Counter()

    entity_count = 0
    zero_candidates = 0
    perfect_entities = 0

    start = time.time()

    with open(args.source1, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            if args.limit and entity_count >= args.limit:
                break

            s1_id = row["entity_id"]

            features = get_features(row)

            candidates, rule_hits = get_candidates(
                conn,
                features
            )

            true_matches = truth.get(s1_id, set())

            retrieved_true = candidates & true_matches

            total_true += len(true_matches)
            total_retrieved_true += len(retrieved_true)
            total_candidates += len(candidates)

            candidate_counts.append(len(candidates))

            if not candidates:
                zero_candidates += 1

            if true_matches and retrieved_true == true_matches:
                perfect_entities += 1

            for rule, count in rule_hits.items():
                rule_total[rule] += count

            entity_count += 1

            if entity_count % 100 == 0:
                elapsed = time.time() - start
                avg = total_candidates / entity_count

                print(
                    f"Processed {entity_count:,} | "
                    f"Avg candidates: {avg:,.2f} | "
                    f"elapsed: {elapsed:.1f}s"
                )

    conn.close()

    candidate_counts.sort()

    def percentile(values, p):
        if not values:
            return 0

        index = int(len(values) * p)
        index = min(index, len(values) - 1)

        return values[index]

    recall = (
        total_retrieved_true / total_true
        if total_true
        else 0
    )

    avg_candidates = (
        total_candidates / entity_count
        if entity_count
        else 0
    )

    reduction_ratio = (
        1 - (
            total_candidates /
            (entity_count * args.db_records)
        )
        if entity_count and args.db_records
        else 0
    )

    print()
    print("=" * 70)
    print("V4 BLOCKING RESULTS")
    print("=" * 70)

    print(f"S1 entities evaluated       : {entity_count:,}")
    print(f"True matches                : {total_true:,}")
    print(f"Retrieved true matches      : {total_retrieved_true:,}")
    print(f"Recall                      : {recall * 100:.4f}%")
    print()
    print(f"Average candidates / S1     : {avg_candidates:,.2f}")
    print(f"Median candidates (P50)     : {percentile(candidate_counts, 0.50):,}")
    print(f"P90 candidates              : {percentile(candidate_counts, 0.90):,}")
    print(f"P95 candidates              : {percentile(candidate_counts, 0.95):,}")
    print(f"P99 candidates              : {percentile(candidate_counts, 0.99):,}")
    print(f"Maximum candidates          : {max(candidate_counts) if candidate_counts else 0:,}")
    print()
    print(f"S1 with zero candidates     : {zero_candidates:,}")
    print(
        f"Zero-candidate rate         : "
        f"{zero_candidates / entity_count * 100:.4f}%"
        if entity_count
        else "Zero-candidate rate         : 0%"
    )
    print()
    print(f"Perfect entity recall       : {perfect_entities:,}")

    print()
    print("PER-RULE CANDIDATE HITS")
    print("-" * 70)

    for rule, count in rule_total.items():
        print(f"{rule:<45} {count:,}")

    print()
    print(f"Evaluation time             : {time.time() - start:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        required=True
    )

    parser.add_argument(
        "--source1",
        required=True
    )

    parser.add_argument(
        "--ground-truth",
        required=True
    )

    parser.add_argument(
        "--db-records",
        type=int,
        required=True
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    evaluate(args)
