import argparse
import csv
import re
import sqlite3
import time
import unicodedata
from difflib import SequenceMatcher
from collections import defaultdict


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
    text = unicodedata.normalize("NFKC", text or "").lower()

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
    return re.sub(r"\s+", " ", text).strip()


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

    return {
        "name_key": " ".join(tokens),
        "sorted_name_key": " ".join(sorted_tokens),
        "token_key": "|".join(sorted_tokens),
        "first_token": tokens[0] if tokens else "",
        "prefix_sig": (
            f"{compact[:4]}:{(len(compact) // 5) * 5}"
            if compact else ""
        ),
        "suffix_sig": (
            f"{compact[-4:]}:{(len(compact) // 5) * 5}"
            if compact else ""
        ),
        "tokens": set(tokens),
    }


def address_features(address):
    normalized = normalize_text(address)

    tokens = [
        x for x in normalized.split()
        if x not in GENERIC_ADDRESS_TOKENS
    ]

    numbers = re.findall(r"\b\d+[a-z]?\b", normalized)

    informative = max(tokens, key=len) if tokens else ""

    return {
        "address_key": normalized,
        "address_number": numbers[0] if numbers else "",
        "address_token_sig": informative[:5],
        "tokens": set(tokens),
    }


def features(row):
    country_raw = (row.get("country") or "").strip()

    return {
        "country": COUNTRY_MAP.get(
            country_raw.upper(),
            country_raw
        ),
        "name": name_features(row.get("business_name", "")),
        "address": address_features(
            row.get("business_address", "")
        ),
    }


def candidate_query(conn, f):
    n = f["name"]
    a = f["address"]
    country = f["country"]

    # One SQL statement.
    # IMPORTANT:
    # This is still V4 logic, just executed together.
    sql = """
    SELECT
        entity_id,
        business_name,
        business_address,
        country
    FROM records
    WHERE country = ?
      AND (
            name_key = ?
         OR sorted_name_key = ?
         OR (
                name_token_key = ?
                AND address_number = ?
            )
         OR (
                first_name_token = ?
                AND address_number = ?
            )
         OR address_key = ?
         OR (
                name_prefix_sig = ?
                AND address_number = ?
            )
         OR (
                name_suffix_sig = ?
                AND address_number = ?
            )
         OR (
                name_prefix_sig = ?
                AND address_token_sig = ?
            )
         OR (
                name_suffix_sig = ?
                AND address_token_sig = ?
            )
         OR (
                first_name_token = ?
                AND address_token_sig = ?
            )
      )
    """

    params = (
        country,

        n["name_key"],
        n["sorted_name_key"],

        n["token_key"],
        a["address_number"],

        n["first_token"],
        a["address_number"],

        a["address_key"],

        n["prefix_sig"],
        a["address_number"],

        n["suffix_sig"],
        a["address_number"],

        n["prefix_sig"],
        a["address_token_sig"],

        n["suffix_sig"],
        a["address_token_sig"],

        n["first_token"],
        a["address_token_sig"],
    )

    return conn.execute(sql, params).fetchall()


def jaccard(a, b):
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def similarity(s1_name, s2_name, s1_addr, s2_addr):
    n1 = name_features(s1_name)
    n2 = name_features(s2_name)

    a1 = address_features(s1_addr)
    a2 = address_features(s2_addr)

    # Strong name signals
    name_exact = (
        n1["name_key"] == n2["name_key"]
    )

    name_sorted = (
        n1["sorted_name_key"] ==
        n2["sorted_name_key"]
    )

    name_ratio = SequenceMatcher(
        None,
        n1["name_key"],
        n2["name_key"]
    ).ratio()

    name_token_j = jaccard(
        n1["tokens"],
        n2["tokens"]
    )

    # Address signals
    addr_exact = (
        a1["address_key"] ==
        a2["address_key"]
    )

    addr_ratio = SequenceMatcher(
        None,
        a1["address_key"],
        a2["address_key"]
    ).ratio()

    addr_token_j = jaccard(
        a1["tokens"],
        a2["tokens"]
    )

    number_match = (
        bool(a1["address_number"])
        and
        a1["address_number"] ==
        a2["address_number"]
    )

    # Weighted score
    score = (
        0.32 * name_ratio
        + 0.18 * name_token_j
        + 0.22 * addr_ratio
        + 0.10 * addr_token_j
        + 0.08 * float(number_match)
        + 0.05 * float(name_sorted)
        + 0.05 * float(name_exact)
    )

    # Exact address is a very strong signal.
    if addr_exact:
        score += 0.20

    # Exact name + same address number is extremely strong.
    if name_exact and number_match:
        score += 0.20

    return min(score, 1.0)


def main(args):
    conn = sqlite3.connect(
        args.db
    )

    conn.execute("PRAGMA cache_size=-200000")
    conn.execute("PRAGMA temp_store=MEMORY")

    source2 = {}
    source3 = {}

    # We don't actually need to load S2/S3 separately because
    # the SQLite DB already contains their records.

    matching_path = args.matching
    candidate_path = args.candidates

    matching_file = open(
        matching_path,
        "w",
        encoding="utf-8",
        newline=""
    )

    candidate_file = open(
        candidate_path,
        "w",
        encoding="utf-8",
        newline=""
    )

    matching_writer = csv.writer(
        matching_file,
        delimiter="\t",
        lineterminator="\n"
    )

    candidate_writer = csv.writer(
        candidate_file,
        delimiter="\t",
        lineterminator="\n"
    )

    # Official format
    matching_writer.writerow([
        "source1_entity_id",
        "matched_entity_ids"
    ])

    candidate_writer.writerow([
        "source1_entity_id",
        "candidate_entity_id"
    ])

    start = time.time()

    processed = 0
    total_candidates = 0
    total_matches = 0

    with open(
        args.source1,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            if args.limit and processed >= args.limit:
                break

            s1_id = row["entity_id"]

            f1 = features(row)

            rows = candidate_query(
                conn,
                f1
            )

            scored = []

            for entity_id, name, address, country in rows:

                score = similarity(
                    row["business_name"],
                    name,
                    row["business_address"],
                    address
                )

                scored.append(
                    (
                        score,
                        entity_id
                    )
                )

            # Candidate file must contain exactly
            # the candidates actually scored.
            for _, entity_id in scored:
                candidate_writer.writerow([
                    s1_id,
                    entity_id
                ])

            total_candidates += len(scored)

            # -------------------------------------------------
            # MATCH DECISION
            # -------------------------------------------------

            matches = []

            for score, entity_id in scored:

                if score >= args.threshold:
                    matches.append(
                        (score, entity_id)
                    )

            # Sort strongest first
            matches.sort(
                reverse=True
            )

            # Don't allow an absurd number of matches.
            if len(matches) > args.max_matches:
                matches = matches[
                    :args.max_matches
                ]

            matched_ids = [
                entity_id
                for _, entity_id in matches
            ]

            total_matches += len(matched_ids)

            matching_writer.writerow([
                s1_id,
                ",".join(matched_ids)
            ])

            processed += 1

            if processed % 1000 == 0:

                elapsed = time.time() - start

                print(
                    f"Processed: {processed:,} | "
                    f"Candidates: {total_candidates:,} | "
                    f"Matches: {total_matches:,} | "
                    f"Avg candidates: "
                    f"{total_candidates / processed:,.2f} | "
                    f"Elapsed: {elapsed / 60:.1f} min"
                )

    matching_file.close()
    candidate_file.close()
    conn.close()

    elapsed = time.time() - start

    print()
    print("=" * 70)
    print("SUBMISSION GENERATION COMPLETE")
    print("=" * 70)
    print(f"S1 processed       : {processed:,}")
    print(f"Total candidates   : {total_candidates:,}")
    print(
        f"Average candidates : "
        f"{total_candidates / processed:,.2f}"
        if processed else
        "Average candidates : 0"
    )
    print(f"Total predicted matches : {total_matches:,}")
    print(f"Time               : {elapsed / 60:.2f} minutes")
    print()
    print(f"Matching file      : {matching_path}")
    print(f"Candidate file     : {candidate_path}")


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
        "--matching",
        required=True
    )

    parser.add_argument(
        "--candidates",
        required=True
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.72
    )

    parser.add_argument(
        "--max-matches",
        type=int,
        default=11
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0
    )

    args = parser.parse_args()

    main(args)
