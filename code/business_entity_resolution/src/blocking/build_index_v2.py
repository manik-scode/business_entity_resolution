import argparse
import csv
import re
import sqlite3
import unicodedata
from pathlib import Path
import time


# ============================================================
# CONFIG
# ============================================================

DEFAULT_DB = "results/blocking/blocking_train_v2.db"

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

    # Indian legal forms
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

# Precompiled translation table and regexes for high performance
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

    text = unicodedata.normalize("NFKC", text).lower().translate(PUNCT_TABLE)

    for pattern, rep in COMPILED_REPLS:
        text = pattern.sub(rep, text)

    return " ".join(text.split())


def name_tokens(name: str):
    normalized = normalize_text(name)

    tokens = normalized.split()

    tokens = [
        token
        for token in tokens
        if token not in IGNORED_NAME_TERMS
    ]

    return tokens


def extract_name_keys(name: str):
    tokens = name_tokens(name)

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


def extract_address_tokens(address: str):
    normalized = normalize_text(address)

    tokens = normalized.split()

    result = []

    for token in tokens:
        token = ADDRESS_ABBREVIATIONS.get(token, token)

        if token in IGNORED_ADDRESS_TOKENS:
            continue

        result.append(token)

    return result


def extract_address_number(address: str):
    normalized = normalize_text(address)

    match = NUM_RE.search(normalized)

    if match:
        return match.group(0)

    return ""


# ============================================================
# V2 BLOCKING SIGNATURES
# ============================================================

def name_prefix(name_key: str):
    """
    First 4 characters of normalized name.
    """

    compact = name_key.replace(" ", "")

    if len(compact) < 4:
        return compact

    return compact[:4]


def name_suffix(name_key: str):
    """
    Last 4 characters of normalized name.
    """

    compact = name_key.replace(" ", "")

    if len(compact) < 4:
        return compact

    return compact[-4:]


def name_length_bucket(name_key: str):
    """
    Bucket names by length.

    Example:
        1-5
        6-10
        11-15
        ...
    """

    length = len(name_key.replace(" ", ""))

    bucket = (length // 5) * 5

    return bucket


def name_prefix_signature(name_key: str):
    prefix = name_prefix(name_key)
    bucket = name_length_bucket(name_key)

    if not prefix:
        return ""

    return f"{prefix}:{bucket}"


def name_suffix_signature(name_key: str):
    suffix = name_suffix(name_key)
    bucket = name_length_bucket(name_key)

    if not suffix:
        return ""

    return f"{suffix}:{bucket}"


def informative_address_token(address: str):
    """
    Select the strongest address token.

    We intentionally avoid generic/common words.
    """

    tokens = extract_address_tokens(address)

    if not tokens:
        return ""

    # Prefer longer tokens because they are generally more informative.
    tokens = sorted(
        set(tokens),
        key=lambda x: (-len(x), x)
    )

    for token in tokens:
        if len(token) >= 5:
            return token

    return tokens[0]


def address_prefix_signature(address: str):
    token = informative_address_token(address)

    if not token:
        return ""

    return token[:5]


# ============================================================
# DATABASE
# ============================================================

def create_database(conn):
    cursor = conn.cursor()

    cursor.executescript(
        """
        DROP TABLE IF EXISTS records;

        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            country TEXT NOT NULL,

            business_name TEXT NOT NULL,
            business_address TEXT NOT NULL,

            name_key TEXT,
            sorted_name_key TEXT,
            name_token_key TEXT,
            first_name_token TEXT,

            address_number TEXT,
            address_key TEXT,

            name_prefix_sig TEXT,
            name_suffix_sig TEXT,
            address_token_sig TEXT
        );
        """
    )

    conn.commit()


CREATE_INDEXES_SQL = """
CREATE INDEX idx_country_name
ON records(country, name_key);

CREATE INDEX idx_country_sorted_name
ON records(country, sorted_name_key);

CREATE INDEX idx_country_token_addr
ON records(country, name_token_key, address_number);

CREATE INDEX idx_country_first_addr
ON records(country, first_name_token, address_number);

CREATE INDEX idx_country_address
ON records(country, address_key);

CREATE INDEX idx_country_prefix
ON records(country, name_prefix_sig);

CREATE INDEX idx_country_suffix
ON records(country, name_suffix_sig);

CREATE INDEX idx_country_addr_token
ON records(country, address_token_sig);
"""


# ============================================================
# INSERT
# ============================================================

INSERT_SQL = """
INSERT INTO records (
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
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def process_file(
    conn,
    file_path: Path,
    source: str,
    batch_size: int = 50_000,
):
    print(f"\nProcessing {source}: {file_path}")

    cursor = conn.cursor()

    batch = []
    count = 0

    with open(
        file_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            entity_id = row["entity_id"]
            business_name = row["business_name"] or ""
            business_address = row["business_address"] or ""
            country = row["country"] or ""

            (
                name_key,
                sorted_name_key,
                name_token_key,
                first_name_token,
            ) = extract_name_keys(business_name)

            address_key = normalize_text(business_address)

            address_number = extract_address_number(address_key)

            prefix_sig = name_prefix_signature(name_key)

            suffix_sig = name_suffix_signature(name_key)

            address_token_sig = address_prefix_signature(address_key)

            batch.append(
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

                    prefix_sig,
                    suffix_sig,
                    address_token_sig,
                )
            )

            count += 1

            if len(batch) >= batch_size:

                cursor.executemany(
                    INSERT_SQL,
                    batch,
                )

                conn.commit()

                batch.clear()

                if count % 500_000 == 0:
                    print(
                        f"{source}: {count:,} rows"
                    )

    if batch:
        cursor.executemany(
            INSERT_SQL,
            batch,
        )

        conn.commit()

    print(
        f"{source}: {count:,} rows indexed"
    )

    return count


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
    )

    parser.add_argument(
        "--data-root",
        default="student_resource/dataset",
    )

    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.mode == "train":

        source_dir = data_root / "train"

        s2_file = source_dir / "train_source2.tsv"
        s3_file = source_dir / "train_source3.tsv"

    else:

        source_dir = data_root / "test"

        s2_file = source_dir / "test_source2.tsv"
        s3_file = source_dir / "test_source3.tsv"

    db_path = Path(args.db)

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION - BLOCKING V2")
    print("=" * 70)

    print(f"Mode       : {args.mode}")
    print(f"Data root  : {data_root}")
    print(f"Database   : {db_path}")

    conn = sqlite3.connect(
        db_path
    )

    conn.execute(
        "PRAGMA journal_mode=WAL"
    )

    conn.execute(
        "PRAGMA synchronous=OFF"
    )

    conn.execute(
        "PRAGMA temp_store=MEMORY"
    )

    conn.execute(
        "PRAGMA cache_size=-131072"
    )

    conn.execute(
        "PRAGMA mmap_size=30000000000"
    )

    create_database(conn)

    total = 0

    total += process_file(
        conn,
        s2_file,
        "S2",
        args.batch_size,
    )

    total += process_file(
        conn,
        s3_file,
        "S3",
        args.batch_size,
    )

    print("\nCreating indexes...")

    conn.executescript(
        CREATE_INDEXES_SQL
    )

    conn.execute(
        "ANALYZE"
    )

    conn.commit()

    size_mb = db_path.stat().st_size / (
        1024 * 1024
    )

    print("\n" + "=" * 70)
    print("V2 BUILD COMPLETE")
    print("=" * 70)

    print(
        f"Total indexed : {total:,}"
    )

    print(
        f"Database size : {size_mb:.2f} MB"
    )

    print(
        f"Database path : {db_path}"
    )

    conn.close()


if __name__ == "__main__":
    main()
