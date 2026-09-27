import argparse
import csv
import re
import sqlite3
import statistics
import time
import unicodedata
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DB = PROJECT_ROOT / "results" / "blocking" / "blocking_test_v2.db"
DEFAULT_S1 = PROJECT_ROOT / "student_resource" / "dataset" / "test" / "test_source1.tsv"


# ============================================================
# LOOKUPS & PRECOMPILED CONSTANTS
# ============================================================

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
    "corp",
    "co",
    "inc",
    "ltd",
    "pvt",
    "pub",
    "llc",
    "llp",
    "plc",
    "sa",
    "sarl",
    "gmbh",
    "ag",
    "limited",
    "company",
    "corporation",
}

IGNORED_ADDRESS_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "lane",
    "ln",
    "avenue",
    "ave",
    "boulevard",
    "blvd",
    "highway",
    "hwy",
    "sector",
    "sec",
    "block",
    "district",
    "dist",
    "city",
    "town",
    "village",
    "near",
    "opposite",
    "opp",
    "india",
    "usa",
    "us",
    "france",
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
# HELPERS
# ============================================================

def percentile(values, p):
    if not values:
        return 0

    values = sorted(values)

    index = (len(values) - 1) * p / 100
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    weight = index - lower

    return values[lower] * (1 - weight) + values[upper] * weight


def get_candidates(cursor, country, column, value):
    """
    Query one V2 blocking key.
    """

    if not value:
        return set()

    sql = f"""
        SELECT entity_id
        FROM records
        WHERE country = ?
          AND {column} = ?
    """

    cursor.execute(sql, (country, value))

    return {row[0] for row in cursor.fetchall()}


# ============================================================
# V2 KEY EXTRACTION
# ============================================================

def normalize_for_key(text):
    """
    Lightweight normalization for test S1 keys.
    Must remain compatible with V2 database keys.
    """
    if not text:
        return ""

    text = (
        unicodedata.normalize("NFKC", text)
        .lower()
        .translate(PUNCT_TABLE)
    )

    for pattern, new in COMPILED_REPLS:
        text = pattern.sub(new, text)

    return " ".join(text.split())


def extract_name_keys(name):
    normalized = normalize_for_key(name)

    tokens = normalized.split()

    tokens = [
        token
        for token in tokens
        if token not in IGNORED_NAME_TERMS
    ]

    if not tokens:
        return "", "", "", ""

    name_key = " ".join(tokens)

    sorted_name_key = " ".join(sorted(tokens))

    token_key = "|".join(sorted(set(tokens)))

    first_token = tokens[0]

    return (
        name_key,
        sorted_name_key,
        token_key,
        first_token,
    )


def extract_address_keys(address):
    normalized = normalize_for_key(address)

    tokens = normalized.split()

    result = []

    for token in tokens:
        token = ADDRESS_ABBREVIATIONS.get(token, token)

        if token in IGNORED_ADDRESS_TOKENS:
            continue

        result.append(token)

    address_key = normalized

    number_match = NUM_RE.search(normalized)

    address_number = (
        number_match.group(0)
        if number_match
        else ""
    )

    return address_key, address_number, result


def name_prefix(name_key):
    compact = name_key.replace(" ", "")

    if len(compact) < 4:
        return compact

    return compact[:4]


def name_suffix(name_key):
    compact = name_key.replace(" ", "")

    if len(compact) < 4:
        return compact

    return compact[-4:]


def name_length_bucket(name_key):
    length = len(name_key.replace(" ", ""))

    return (length // 5) * 5


def name_prefix_signature(name_key):
    prefix = name_prefix(name_key)

    if not prefix:
        return ""

    return f"{prefix}:{name_length_bucket(name_key)}"


def name_suffix_signature(name_key):
    suffix = name_suffix(name_key)

    if not suffix:
        return ""

    return f"{suffix}:{name_length_bucket(name_key)}"


def informative_address_token(address):
    _, _, tokens = extract_address_keys(address)

    if not tokens:
        return ""

    tokens = sorted(
        set(tokens),
        key=lambda x: (-len(x), x)
    )

    for token in tokens:
        if len(token) >= 5:
            return token

    return tokens[0]


def address_token_signature(address):
    token = informative_address_token(address)

    if not token:
        return ""

    return token[:5]


# ============================================================
# V2 CANDIDATE GENERATION
# ============================================================

COUNTRY_MAP = {
    "INDIA": "India",
    "US": "US",
    "FRANCE": "France",
}


def generate_candidates(cursor, row):
    country_raw = (row.get("country") or "").strip()

    country = COUNTRY_MAP.get(
        country_raw.upper(),
        country_raw
    )

    name = row.get("business_name") or ""
    address = row.get("business_address") or ""

    (
        name_key,
        sorted_name_key,
        name_token_key,
        first_name_token,
    ) = extract_name_keys(name)

    address_key, address_number, _ = extract_address_keys(address)

    prefix_sig = name_prefix_signature(name_key)

    suffix_sig = name_suffix_signature(name_key)

    address_token_sig = address_token_signature(address)

    candidates = set()

    rule_counts = {}

    # --------------------------------------------------------
    # RULE 1: Exact normalized name
    # --------------------------------------------------------
    result = get_candidates(
        cursor,
        country,
        "name_key",
        name_key
    )
    candidates.update(result)
    rule_counts["R1_exact_name"] = len(result)

    # --------------------------------------------------------
    # RULE 2: Word-order invariant name
    # --------------------------------------------------------
    result = get_candidates(
        cursor,
        country,
        "sorted_name_key",
        sorted_name_key
    )
    candidates.update(result)
    rule_counts["R2_sorted_name"] = len(result)

    # --------------------------------------------------------
    # RULE 3: Name token + address number
    # --------------------------------------------------------
    result = set()
    if name_token_key and address_number:
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
            )
        )
        result = {row[0] for row in cursor.fetchall()}
    candidates.update(result)
    rule_counts["R3_token_address_number"] = len(result)

    # --------------------------------------------------------
    # RULE 4: First name token + address number
    # --------------------------------------------------------
    result = set()
    if first_name_token and address_number:
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
            )
        )
        result = {row[0] for row in cursor.fetchall()}
    candidates.update(result)
    rule_counts["R4_first_token_address_number"] = len(result)

    # --------------------------------------------------------
    # RULE 5: Exact normalized address
    # --------------------------------------------------------
    result = get_candidates(
        cursor,
        country,
        "address_key",
        address_key
    )
    candidates.update(result)
    rule_counts["R5_exact_address"] = len(result)

    # --------------------------------------------------------
    # RULE 6: Name prefix + length bucket
    # --------------------------------------------------------
    result = get_candidates(
        cursor,
        country,
        "name_prefix_sig",
        prefix_sig
    )
    candidates.update(result)
    rule_counts["R6_prefix_length"] = len(result)

    # --------------------------------------------------------
    # RULE 7: Name suffix + length bucket
    # --------------------------------------------------------
    result = get_candidates(
        cursor,
        country,
        "name_suffix_sig",
        suffix_sig
    )
    candidates.update(result)
    rule_counts["R7_suffix_length"] = len(result)

    # --------------------------------------------------------
    # RULE 8: Informative address token
    # --------------------------------------------------------
    result = get_candidates(
        cursor,
        country,
        "address_token_sig",
        address_token_sig
    )
    candidates.update(result)
    rule_counts["R8_address_token"] = len(result)

    return candidates, rule_counts


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB
    )

    parser.add_argument(
        "--s1",
        type=Path,
        default=DEFAULT_S1
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TEST V2 BLOCKING BENCHMARK")
    print("=" * 70)

    print(f"S1 file : {args.s1}")
    print(f"DB      : {args.db}")
    print(f"Limit   : {args.limit:,}")

    if not args.db.exists():
        raise FileNotFoundError(
            f"Database not found: {args.db}"
        )

    if not args.s1.exists():
        raise FileNotFoundError(
            f"S1 file not found: {args.s1}"
        )

    conn = sqlite3.connect(
        f"file:{args.db}?mode=ro",
        uri=True
    )
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA cache_size=-131072")
    conn.execute("PRAGMA mmap_size=30000000000")

    cursor = conn.cursor()

    candidate_counts = []

    total_candidates = 0

    zero_candidates = 0

    processed = 0

    total_time_start = time.time()

    rule_totals = {
        "R1_exact_name": 0,
        "R2_sorted_name": 0,
        "R3_token_address_number": 0,
        "R4_first_token_address_number": 0,
        "R5_exact_address": 0,
        "R6_prefix_length": 0,
        "R7_suffix_length": 0,
        "R8_address_token": 0,
    }

    with open(
        args.s1,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            candidates, rule_counts = generate_candidates(
                cursor,
                row
            )

            count = len(candidates)

            candidate_counts.append(count)

            total_candidates += count

            if count == 0:
                zero_candidates += 1

            for rule, value in rule_counts.items():
                rule_totals[rule] += value

            processed += 1

            if processed % 100 == 0:
                elapsed = time.time() - total_time_start

                print(
                    f"Processed {processed:,} "
                    f"| Avg candidates: "
                    f"{total_candidates / processed:,.2f} "
                    f"| elapsed: {elapsed:.1f}s"
                )

            if processed >= args.limit:
                break

    elapsed = time.time() - total_time_start

    # ========================================================
    # RESULTS
    # ========================================================

    average = (
        total_candidates / processed
        if processed
        else 0
    )

    print()
    print("=" * 70)
    print("V2 TEST BLOCKING RESULTS")
    print("=" * 70)

    print(
        f"S1 entities evaluated       : {processed:,}"
    )

    print(
        f"Total candidates            : {total_candidates:,}"
    )

    print(
        f"Average candidates / S1     : {average:,.2f}"
    )

    print(
        f"Median candidates (P50)     : "
        f"{percentile(candidate_counts, 50):,.0f}"
    )

    print(
        f"P90 candidates              : "
        f"{percentile(candidate_counts, 90):,.0f}"
    )

    print(
        f"P95 candidates              : "
        f"{percentile(candidate_counts, 95):,.0f}"
    )

    print(
        f"P99 candidates              : "
        f"{percentile(candidate_counts, 99):,.0f}"
    )

    print(
        f"Maximum candidates          : "
        f"{max(candidate_counts) if candidate_counts else 0:,}"
    )

    print(
        f"S1 with zero candidates     : "
        f"{zero_candidates:,}"
    )

    print(
        f"Zero-candidate rate         : "
        f"{(zero_candidates / processed * 100) if processed else 0:.4f}%"
    )

    print()
    print("-" * 70)
    print("PER-RULE RAW CANDIDATE HITS")
    print("-" * 70)

    for rule, count in rule_totals.items():
        print(
            f"{rule:<38} {count:>12,}"
        )

    print()
    print(
        f"Evaluation time             : {elapsed:.2f}s"
    )

    conn.close()


if __name__ == "__main__":
    main()
