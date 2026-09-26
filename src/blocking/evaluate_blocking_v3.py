import argparse
import csv
import sqlite3
import statistics
import unicodedata
import re
import time
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

DEFAULT_DB = "results/blocking/blocking_train_v2.db"

# Maximum number of records allowed for broad blocking rules.
PREFIX_MAX_FREQ = 2000
SUFFIX_MAX_FREQ = 2000
ADDRESS_TOKEN_MAX_FREQ = 1000


NAME_REPLACEMENTS = {
    "corporation": "corp",
    "company": "co",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "public": "pub",
    "societe": "soc",
    "societe anonyme": "sa",
    "etablissement": "etb",
    "private limited": "pvt ltd",
    "public limited": "pub ltd",
}


IGNORED_NAME_TERMS = {
    "corp", "co", "inc", "ltd", "pvt", "pub",
    "llc", "llp", "plc", "sa", "sarl", "gmbh",
    "ag", "limited", "company", "corporation",
}


IGNORED_ADDRESS_TOKENS = {
    "road", "rd", "street", "st", "lane", "ln",
    "avenue", "ave", "boulevard", "blvd",
    "highway", "hwy", "sector", "sec", "block",
    "district", "dist", "city", "town", "village",
    "near", "opposite", "opp", "india", "usa",
    "us", "france",
}


ADDRESS_ABBREVIATIONS = {
    "road": "rd",
    "street": "st",
    "lane": "ln",
    "avenue": "ave",
    "boulevard": "blvd",
    "highway": "hwy",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
}


PUNCT_TABLE = {
    cp: " "
    for cp in range(0x10000)
    if unicodedata.category(chr(cp)).startswith(("P", "S"))
}


COMPILED_REPLS = [
    (re.compile(rf"\b{re.escape(k)}\b"), v)
    for k, v in NAME_REPLACEMENTS.items()
]


NUM_RE = re.compile(r"\b\d+[a-z]?\b")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = (
        unicodedata
        .normalize("NFKC", text)
        .lower()
        .translate(PUNCT_TABLE)
    )

    for pattern, replacement in COMPILED_REPLS:
        text = pattern.sub(replacement, text)

    return " ".join(text.split())


def extract_name_keys(name: str):

    normalized = normalize_text(name)

    tokens = [
        token
        for token in normalized.split()
        if token not in IGNORED_NAME_TERMS
    ]

    if not tokens:
        return "", "", "", ""

    name_key = " ".join(tokens)

    sorted_name_key = " ".join(
        sorted(tokens)
    )

    token_key = "|".join(
        sorted(set(tokens))
    )

    first_token = tokens[0]

    return (
        name_key,
        sorted_name_key,
        token_key,
        first_token,
    )


def extract_address_number(address: str):

    match = NUM_RE.search(address)

    if match:
        return match.group(0)

    return ""


def extract_address_tokens(address: str):

    tokens = []

    for token in address.split():

        token = ADDRESS_ABBREVIATIONS.get(
            token,
            token,
        )

        if token in IGNORED_ADDRESS_TOKENS:
            continue

        tokens.append(token)

    return tokens


def name_prefix_signature(name_key: str):

    compact = name_key.replace(" ", "")

    if not compact:
        return ""

    prefix = compact[:4]

    length = len(compact)
    bucket = (length // 5) * 5

    return f"{prefix}:{bucket}"


def name_suffix_signature(name_key: str):

    compact = name_key.replace(" ", "")

    if not compact:
        return ""

    suffix = compact[-4:]

    length = len(compact)
    bucket = (length // 5) * 5

    return f"{suffix}:{bucket}"


def address_prefix_signature(address: str):

    tokens = extract_address_tokens(address)

    if not tokens:
        return ""

    tokens = sorted(
        set(tokens),
        key=lambda x: (-len(x), x),
    )

    for token in tokens:

        if len(token) >= 5:
            return token[:5]

    return tokens[0]


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(ground_truth_file):

    truth = {}

    with open(
        ground_truth_file,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            s1_id = row["source1_entity_id"]

            raw_matches = (
                row["matched_entity_ids"] or ""
            )

            matches = {
                x.strip()
                for x in raw_matches.split(",")
                if x.strip()
            }

            truth[s1_id] = matches

    return truth


# ============================================================
# SELECTIVE BROAD QUERY
# ============================================================

def selective_query(
    cursor,
    sql,
    params,
    max_frequency,
):
    """
    Return candidates only when the blocking key is selective.

    We request max_frequency + 1 rows.

    If more than max_frequency records exist,
    the rule is considered too broad and returns an empty set.
    """

    cursor.execute(
        sql + f" LIMIT {max_frequency + 1}",
        params,
    )

    rows = cursor.fetchall()

    if len(rows) > max_frequency:
        return set(), False

    return {row[0] for row in rows}, True


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    db_path,
    s1_file,
    ground_truth_file,
    limit,
):

    print("=" * 70)
    print("BLOCKING V3 EVALUATION")
    print("=" * 70)

    print(f"Database : {db_path}")
    print(f"S1 file  : {s1_file}")
    print(f"Limit    : {limit:,}")

    print()
    print("V3 frequency limits:")
    print(f"  Prefix       <= {PREFIX_MAX_FREQ:,}")
    print(f"  Suffix       <= {SUFFIX_MAX_FREQ:,}")
    print(f"  Address token<= {ADDRESS_TOKEN_MAX_FREQ:,}")
    print()

    truth = load_ground_truth(
        ground_truth_file,
    )

    print(
        f"Ground truth loaded: {len(truth):,}"
    )

    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
    )

    conn.execute(
        "PRAGMA query_only=ON"
    )

    conn.execute(
        "PRAGMA cache_size=-131072"
    )

    conn.execute(
        "PRAGMA mmap_size=30000000000"
    )

    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) FROM records"
    )

    indexed_records = cursor.fetchone()[0]

    print(
        f"Indexed records: {indexed_records:,}"
    )

    # ========================================================
    # METRICS
    # ========================================================

    total_true_matches = 0
    retrieved_true_matches = 0
    total_candidates = 0

    zero_candidates = 0

    candidate_counts = []

    perfect_entity_recall = 0
    entities_with_match = 0

    rule_hits = [0] * 8
    rule_unique_hits = [0] * 8

    rule_skipped = [0] * 8

    entity_stats = {}

    start_time = time.time()

    # ========================================================
    # S1
    # ========================================================

    with open(
        s1_file,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for i, row in enumerate(reader):

            if limit and i >= limit:
                break

            s1_id = row["entity_id"]

            true_matches = truth.get(
                s1_id,
                set(),
            )

            country = row["country"]

            business_name = (
                row["business_name"] or ""
            )

            business_address = (
                row["business_address"] or ""
            )

            (
                name_key,
                sorted_name_key,
                name_token_key,
                first_name_token,
            ) = extract_name_keys(
                business_name
            )

            address_key = normalize_text(
                business_address
            )

            address_number = (
                extract_address_number(
                    address_key
                )
            )

            prefix_sig = (
                name_prefix_signature(
                    name_key
                )
            )

            suffix_sig = (
                name_suffix_signature(
                    name_key
                )
            )

            address_token_sig = (
                address_prefix_signature(
                    address_key
                )
            )

            # ------------------------------------------------
            # Candidate sets
            # ------------------------------------------------

            rule_sets = [set() for _ in range(8)]

            # ------------------------------------------------
            # Rule 1
            # Exact normalized name
            # ------------------------------------------------

            if name_key:

                cursor.execute(
                    """
                    SELECT entity_id
                    FROM records
                    WHERE country = ?
                    AND name_key = ?
                    """,
                    (
                        country,
                        name_key,
                    ),
                )

                rule_sets[0] = {
                    x[0]
                    for x in cursor.fetchall()
                }

            # ------------------------------------------------
            # Rule 2
            # Word-order invariant name
            # ------------------------------------------------

            if sorted_name_key:

                cursor.execute(
                    """
                    SELECT entity_id
                    FROM records
                    WHERE country = ?
                    AND sorted_name_key = ?
                    """,
                    (
                        country,
                        sorted_name_key,
                    ),
                )

                rule_sets[1] = {
                    x[0]
                    for x in cursor.fetchall()
                }

            # ------------------------------------------------
            # Rule 3
            # Name tokens + address number
            # ------------------------------------------------

            if (
                name_token_key
                and address_number
            ):

                cursor.execute(
                    """
                    SELECT entity_id
                    FROM records
                    WHERE country = ?
                    AND name_token_key = ?
                    AND address_number = ?
                    """,
                    (
                        country,
                        name_token_key,
                        address_number,
                    ),
                )

                rule_sets[2] = {
                    x[0]
                    for x in cursor.fetchall()
                }

            # ------------------------------------------------
            # Rule 4
            # First name token + address number
            # ------------------------------------------------

            if (
                first_name_token
                and address_number
            ):

                cursor.execute(
                    """
                    SELECT entity_id
                    FROM records
                    WHERE country = ?
                    AND first_name_token = ?
                    AND address_number = ?
                    """,
                    (
                        country,
                        first_name_token,
                        address_number,
                    ),
                )

                rule_sets[3] = {
                    x[0]
                    for x in cursor.fetchall()
                }

            # ------------------------------------------------
            # Rule 5
            # Exact normalized address
            # ------------------------------------------------

            if address_key:

                cursor.execute(
                    """
                    SELECT entity_id
                    FROM records
                    WHERE country = ?
                    AND address_key = ?
                    """,
                    (
                        country,
                        address_key,
                    ),
                )

                rule_sets[4] = {
                    x[0]
                    for x in cursor.fetchall()
                }

            # ------------------------------------------------
            # Rule 6
            # Prefix + length bucket
            # SELECTIVE
            # ------------------------------------------------

            if prefix_sig:

                result, accepted = selective_query(
                    cursor,
                    """
                    SELECT entity_id
                    FROM records
                    WHERE country = ?
                    AND name_prefix_sig = ?
                    """,
                    (
                        country,
                        prefix_sig,
                    ),
                    PREFIX_MAX_FREQ,
                )

                if accepted:
                    rule_sets[5] = result
                else:
                    rule_skipped[5] += 1

            # ------------------------------------------------
            # Rule 7
            # Suffix + length bucket
            # SELECTIVE
            # ------------------------------------------------

            if suffix_sig:

                result, accepted = selective_query(
                    cursor,
                    """
                    SELECT entity_id
                    FROM records
                    WHERE country = ?
                    AND name_suffix_sig = ?
                    """,
                    (
                        country,
                        suffix_sig,
                    ),
                    SUFFIX_MAX_FREQ,
                )

                if accepted:
                    rule_sets[6] = result
                else:
                    rule_skipped[6] += 1

            # ------------------------------------------------
            # Rule 8
            # Address token
            # SELECTIVE
            # ------------------------------------------------

            if address_token_sig:

                result, accepted = selective_query(
                    cursor,
                    """
                    SELECT entity_id
                    FROM records
                    WHERE country = ?
                    AND address_token_sig = ?
                    """,
                    (
                        country,
                        address_token_sig,
                    ),
                    ADDRESS_TOKEN_MAX_FREQ,
                )

                if accepted:
                    rule_sets[7] = result
                else:
                    rule_skipped[7] += 1

            # ------------------------------------------------
            # UNION
            # ------------------------------------------------

            candidates = set().union(
                *rule_sets
            )

            candidate_count = len(
                candidates
            )

            candidate_counts.append(
                candidate_count
            )

            total_candidates += (
                candidate_count
            )

            if candidate_count == 0:
                zero_candidates += 1

            # ------------------------------------------------
            # Recall
            # ------------------------------------------------

            retrieved = (
                true_matches
                & candidates
            )

            total_true_matches += len(
                true_matches
            )

            retrieved_true_matches += len(
                retrieved
            )

            # ------------------------------------------------
            # Entity statistics
            # ------------------------------------------------

            match_count = len(
                true_matches
            )

            if match_count not in entity_stats:

                entity_stats[
                    match_count
                ] = {
                    "entities": 0,
                    "true": 0,
                    "found": 0,
                }

            entity_stats[
                match_count
            ]["entities"] += 1

            entity_stats[
                match_count
            ]["true"] += match_count

            entity_stats[
                match_count
            ]["found"] += len(
                retrieved
            )

            if len(true_matches) > 0:

                entities_with_match += 1

                if (
                    len(retrieved)
                    == len(true_matches)
                ):
                    perfect_entity_recall += 1

            # ------------------------------------------------
            # Rule attribution
            # ------------------------------------------------

            seen = set()

            for idx, rule_result in enumerate(
                rule_sets
            ):

                hits = (
                    true_matches
                    & rule_result
                )

                rule_hits[idx] += len(
                    hits
                )

                unique_hits = (
                    hits - seen
                )

                rule_unique_hits[idx] += len(
                    unique_hits
                )

                seen.update(
                    rule_result
                )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                (i + 1) % 1000 == 0
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                rate = (
                    (i + 1) / elapsed
                    if elapsed > 0
                    else 0
                )

                current_recall = (
                    retrieved_true_matches
                    / total_true_matches
                    * 100
                    if total_true_matches
                    else 0
                )

                avg_now = (
                    total_candidates
                    / (i + 1)
                )

                print(
                    f"Evaluated: {i + 1:,} | "
                    f"{rate:,.1f} S1/s | "
                    f"Recall: {current_recall:.2f}% | "
                    f"Avg candidates: {avg_now:,.1f}"
                )

    conn.close()

    # ========================================================
    # FINAL METRICS
    # ========================================================

    recall = (
        retrieved_true_matches
        / total_true_matches
        * 100
        if total_true_matches
        else 0
    )

    avg_candidates = (
        statistics.mean(
            candidate_counts
        )
        if candidate_counts
        else 0
    )

    sorted_candidates = sorted(
        candidate_counts
    )

    def percentile(p):

        if not sorted_candidates:
            return 0

        index = int(
            p * (
                len(sorted_candidates) - 1
            )
        )

        return sorted_candidates[index]

    pairs_quality = (
        retrieved_true_matches
        / total_candidates
        * 100
        if total_candidates
        else 0
    )

    reduction_ratio = (
        1
        -
        (
            total_candidates
            /
            (
                len(candidate_counts)
                * indexed_records
            )
        )
    ) * 100

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BLOCKING V3 RESULTS")
    print("=" * 70)

    print(
        f"S1 entities evaluated       : "
        f"{len(candidate_counts):,}"
    )

    print(
        f"Total indexed in database   : "
        f"{indexed_records:,}"
    )

    print(
        f"True matches                 : "
        f"{total_true_matches:,}"
    )

    print(
        f"True matches retrieved      : "
        f"{retrieved_true_matches:,}"
    )

    print()

    print(
        f"BLOCKING RECALL             : "
        f"{recall:.4f}%"
    )

    print(
        f"Pairs Quality               : "
        f"{pairs_quality:.4f}%"
    )

    print(
        f"Reduction Ratio             : "
        f"{reduction_ratio:.6f}%"
    )

    print()

    print(
        f"Average candidates / S1     : "
        f"{avg_candidates:.2f}"
    )

    print(
        f"Median candidates (P50)     : "
        f"{percentile(0.50):,}"
    )

    print(
        f"P90 candidates              : "
        f"{percentile(0.90):,}"
    )

    print(
        f"P95 candidates              : "
        f"{percentile(0.95):,}"
    )

    print(
        f"P99 candidates              : "
        f"{percentile(0.99):,}"
    )

    print(
        f"Maximum candidates          : "
        f"{max(candidate_counts):,}"
    )

    print(
        f"S1 with zero candidates     : "
        f"{zero_candidates:,}"
    )

    print(
        f"Zero-candidate rate         : "
        f"{zero_candidates / len(candidate_counts) * 100:.4f}%"
    )

    # ========================================================
    # RULE RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("PER-RULE TRUE MATCH HITS")
    print("=" * 70)

    rule_names = [
        "Rule 1 Exact normalized name",
        "Rule 2 Word-order invariant name",
        "Rule 3 Name token + address number",
        "Rule 4 First name token + address number",
        "Rule 5 Exact normalized address",
        "Rule 6 Name prefix + length bucket",
        "Rule 7 Name suffix + length bucket",
        "Rule 8 Informative address token",
    ]

    for i, name in enumerate(
        rule_names
    ):

        print(
            f"{name:<48}"
            f"Hits {rule_hits[i]:>10,}"
            f"  Unique {rule_unique_hits[i]:>10,}"
            f"  Skipped {rule_skipped[i]:>8,}"
        )

    # ========================================================
    # ENTITY RECALL
    # ========================================================

    print()
    print("=" * 70)
    print("ENTITY-LEVEL RECALL")
    print("=" * 70)

    print(
        f"{'Matches':>8}"
        f"{'Entities':>12}"
        f"{'True':>12}"
        f"{'Found':>12}"
        f"{'Recall':>12}"
    )

    for match_count in sorted(
        entity_stats
    ):

        stats = entity_stats[
            match_count
        ]

        entity_recall = (
            stats["found"]
            / stats["true"]
            * 100
            if stats["true"]
            else 0
        )

        print(
            f"{match_count:>8}"
            f"{stats['entities']:>12,}"
            f"{stats['true']:>12,}"
            f"{stats['found']:>12,}"
            f"{entity_recall:>11.2f}%"
        )

    print()

    print(
        f"Entities with >=1 true match : "
        f"{entities_with_match:,}"
    )

    print(
        f"Perfect entity recall         : "
        f"{perfect_entity_recall:,}"
    )

    elapsed_total = (
        time.time() - start_time
    )

    print(
        f"\nEvaluation completed in "
        f"{elapsed_total:.1f}s "
        f"({elapsed_total / 60:.2f} min)"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1_000,
    )

    parser.add_argument(
        "--data-root",
        default="student_resource/dataset",
    )

    args = parser.parse_args()

    data_root = Path(
        args.data_root
    )

    s1_file = (
        data_root
        / "train"
        / "train_source1.tsv"
    )

    ground_truth_file = (
        data_root
        / "train"
        / "train_ground_truth.tsv"
    )

    evaluate(
        args.db,
        s1_file,
        ground_truth_file,
        args.limit,
    )


if __name__ == "__main__":
    main()
