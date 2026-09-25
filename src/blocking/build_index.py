import argparse
import csv
from pathlib import Path
import re
import sqlite3
import time
import unicodedata

# =========================================================
# PATH CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "student_resource" / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results" / "blocking"


# =========================================================
# REGEX & LOOKUP CONSTANTS (Precompiled for high performance)
# =========================================================

# Build Unicode punctuation & symbol translation table (maps all P* and S* chars to spaces)
# This avoids the catastrophic bug in re.sub(r'[^\w\s]') where Python's \w drops
# Unicode combining marks (matras and viramas) and destroys Indic/South Asian words.
# Running in C via str.translate is also ~5x faster than regex re.sub in Python.
PUNCT_SYMBOL_TABLE = {
    codepoint: " "
    for codepoint in range(0x10000)
    if unicodedata.category(chr(codepoint))[0] in ("P", "S")
}

NUM_RE = re.compile(r"\b\d+[a-z]?\b")

# Legal entity suffixes to normalize across English, French, and Indic scripts
NAME_REPLACEMENTS = {
    # English
    "corporation": "corp",
    "company": "co",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "public": "pub",
    # French
    "societe": "soc",
    "etablissement": "etb",
    # Indic (Devanagari)
    "प्राइवेट": "pvt",
    "लिमिटेड": "ltd",
    "कंपनी": "co",
    "प्रा": "pvt",
    "लि": "ltd",
}

# Legal suffixes to exclude from distinctive token keys
IGNORED_LEGAL = {
    "inc",
    "corp",
    "co",
    "ltd",
    "pvt",
    "llc",
    "llp",
    "limited",
    "company",
    "corporation",
    "incorporated",
    "private",
    "plc",
    "gmbh",
    "sarl",
    "sas",
    "sa",
    "soc",
    "etb",
}

# Generic business terms to exclude from distinctive token keys
GENERIC_BUSINESS_TERMS = {
    "services",
    "service",
    "solutions",
    "solution",
    "technologies",
    "technology",
    "enterprises",
    "enterprise",
    "consulting",
    "consultants",
    "management",
    "holdings",
    "holding",
    "group",
    "international",
    "global",
    "industries",
    "industry",
    "associates",
    "foundation",
    "ventures",
    "venture",
    "commercial",
    "commercials",
    "trading",
    "products",
    "systems",
}

ALL_STOPWORDS = IGNORED_LEGAL | GENERIC_BUSINESS_TERMS

# Common address abbreviation normalizations
ADDRESS_REPLACEMENTS = {
    "road": "rd",
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "highway": "hwy",
    "apartment": "apt",
    "suite": "ste",
    "building": "bldg",
    "floor": "fl",
}


# =========================================================
# DATABASE SCHEMAS
# =========================================================

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS records (
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
    address_key TEXT
);
"""

# Composite indexes designed for high recall and tight selectivity
CREATE_INDEX_SQL = [
    # 1. Exact normalized name match within country
    """
    CREATE INDEX IF NOT EXISTS idx_country_name
    ON records(country, name_key);
    """,

    # 2. Word-order invariant name match within country
    """
    CREATE INDEX IF NOT EXISTS idx_country_sorted_name
    ON records(country, sorted_name_key);
    """,

    # 3. Distinctive name token + building number (also covers queries on (country, name_token_key))
    """
    CREATE INDEX IF NOT EXISTS idx_country_token_addr
    ON records(country, name_token_key, address_number);
    """,

    # 4. First name token + building number (handles brand lead + address match)
    """
    CREATE INDEX IF NOT EXISTS idx_country_first_addr
    ON records(country, first_name_token, address_number);
    """,

    # 5. Normalized address match (handles trade names / DBA where name changed)
    """
    CREATE INDEX IF NOT EXISTS idx_country_address
    ON records(country, address_key);
    """,
]

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
    address_key
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


# =========================================================
# FEATURE EXTRACTION & NORMALIZATION (OPTIMIZED SINGLE-PASS)
# =========================================================

def normalize_text(value: str) -> str:
    """
    Normalize unicode, lowercase, and replace punctuation & symbols with spaces.
    Preserves all letters, numbers, and Unicode combining marks (matras, viramas).
    """
    if not value:
        return ""
    # Unicode NFKC normalization followed by translate table lookup
    return unicodedata.normalize("NFKC", value).lower().translate(PUNCT_SYMBOL_TABLE)


def extract_name_keys(raw_name: str):
    """
    Single-pass extraction of all name-based blocking keys.
    5x faster than repeatedly normalizing and regex-processing the raw string.
    Returns:
        (name_key, sorted_name_key, name_token_key, first_name_token)
    """
    if not raw_name:
        return "", "", "", ""

    cleaned = normalize_text(raw_name)
    raw_tokens = cleaned.split()
    if not raw_tokens:
        return "", "", "", ""

    # Normalize legal abbreviations
    tokens = [NAME_REPLACEMENTS.get(tok, tok) for tok in raw_tokens]

    name_key = " ".join(tokens)
    sorted_name_key = " ".join(sorted(tokens))

    # Distinctive name token: pick longest token not in stopwords
    candidate_tokens = [tok for tok in tokens if tok not in ALL_STOPWORDS and len(tok) >= 3]

    if candidate_tokens:
        name_token_key = max(candidate_tokens, key=len)
    else:
        # Fallback 1: non-legal tokens of any length (e.g. short acronyms like IBM, DHL, BMW, EY)
        non_legal = [tok for tok in tokens if tok not in IGNORED_LEGAL]
        if non_legal:
            name_token_key = max(non_legal, key=len)
        else:
            name_token_key = tokens[0]

    # First significant name token (brand or company lead word)
    first_name_token = ""
    for tok in tokens:
        if tok not in IGNORED_LEGAL:
            first_name_token = tok
            break
    if not first_name_token:
        first_name_token = tokens[0]

    return name_key, sorted_name_key, name_token_key, first_name_token


def extract_address_keys(raw_address: str):
    """
    Extracts building number and normalized address key.
    Returns:
        (address_number, address_key)
    """
    if not raw_address:
        return "", ""

    cleaned = normalize_text(raw_address)
    tokens = cleaned.split()
    if not tokens:
        return "", ""

    # First numeric/alphanumeric building number (e.g., 100, 4b, 15208)
    num_match = NUM_RE.search(cleaned)
    address_number = num_match.group(0) if num_match else ""

    norm_tokens = [ADDRESS_REPLACEMENTS.get(tok, tok) for tok in tokens]
    address_key = " ".join(norm_tokens)

    return address_number, address_key


# =========================================================
# DATABASE SETUP
# =========================================================

def init_database(db_path: Path):
    """Initializes SQLite database with performance-tuned PRAGMAs."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        print(f"Removing existing database: {db_path}")
        db_path.unlink()

    connection = sqlite3.connect(db_path)

    # Performance PRAGMAs for high-throughput bulk inserts and index builds
    connection.execute("PRAGMA journal_mode = WAL;")
    connection.execute("PRAGMA synchronous = OFF;")
    connection.execute("PRAGMA temp_store = MEMORY;")
    connection.execute("PRAGMA cache_size = -131072;")  # ~128MB cache
    connection.execute("PRAGMA mmap_size = 30000000000;")  # 30GB mmap

    connection.execute(CREATE_TABLE_SQL)
    connection.commit()

    return connection


# =========================================================
# PROCESS SOURCE
# =========================================================

def process_source(connection, source_name: str, path: Path, batch_size: int = 50_000, limit: int = None):
    """Reads TSV source file and bulk inserts extracted keys into database."""
    print()
    print("=" * 70)
    print(f"BUILDING RECORDS: {source_name}")
    print("=" * 70)
    print(f"File: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    rows = []
    total = 0
    t0 = time.time()
    t_batch = t0

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            entity_id = row["entity_id"]
            business_name = row.get("business_name", "") or ""
            business_address = row.get("business_address", "") or ""
            country = (row.get("country", "") or "").strip().upper()

            # Single-pass key generation
            name_key, sorted_name_key, name_token_key, first_name_token = extract_name_keys(business_name)
            address_number, address_key = extract_address_keys(business_address)

            rows.append((
                entity_id,
                source_name,
                country,
                business_name,
                business_address,
                name_key,
                sorted_name_key,
                name_token_key,
                first_name_token,
                address_number,
                address_key,
            ))

            total += 1

            if len(rows) >= batch_size:
                connection.executemany(INSERT_SQL, rows)
                connection.commit()
                rows.clear()

                if total % 500_000 == 0:
                    now = time.time()
                    rate = 500_000 / (now - t_batch) if (now - t_batch) > 0 else 0
                    print(f"  Processed {total:,} rows ({rate:,.0f} rows/s, elapsed: {now - t0:.1f}s)")
                    t_batch = now

            if limit and total >= limit:
                print(f"  Reached limit of {limit:,} rows. Stopping.")
                break

    if rows:
        connection.executemany(INSERT_SQL, rows)
        connection.commit()

    total_time = time.time() - t0
    avg_rate = total / total_time if total_time > 0 else 0
    print(f"Finished {source_name}: {total:,} rows in {total_time:.1f}s ({avg_rate:,.0f} rows/s)")


# =========================================================
# CREATE INDEXES
# =========================================================

def create_indexes(connection):
    """Builds indexes in a single pass after bulk data load."""
    print()
    print("=" * 70)
    print("CREATING SQLITE INDEXES")
    print("=" * 70)

    for i, sql in enumerate(CREATE_INDEX_SQL, start=1):
        idx_name = sql.strip().split()[5] if len(sql.strip().split()) > 5 else f"index_{i}"
        t0 = time.time()
        print(f"[{i}/{len(CREATE_INDEX_SQL)}] Creating {idx_name}...")
        connection.execute(sql)
        connection.commit()
        print(f"      Completed in {time.time() - t0:.2f}s")


# =========================================================
# DATABASE SUMMARY
# =========================================================

def print_summary(connection, db_path: Path):
    """Prints indexed record counts and database footprint."""
    print()
    print("=" * 70)
    print("DATABASE SUMMARY")
    print("=" * 70)

    cursor = connection.execute("SELECT COUNT(*) FROM records")
    total = cursor.fetchone()[0]
    print(f"Total indexed records: {total:,}")

    cursor = connection.execute("""
        SELECT source, country, COUNT(*) 
        FROM records 
        GROUP BY source, country 
        ORDER BY source, country
    """)
    for source, country, count in cursor.fetchall():
        print(f"  {source} [{country}]: {count:,}")

    print()
    print(f"Database location: {db_path}")
    if db_path.exists():
        size_gb = db_path.stat().st_size / (1024 ** 3)
        print(f"Database file size: {size_gb:.2f} GB ({db_path.stat().st_size / (1024 ** 2):.1f} MB)")


# =========================================================
# CLI & MAIN
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build SQLite Blocking Index for Business Entity Resolution"
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Dataset split to index (train or test). Default: train",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Path to dataset root folder. Default: student_resource/dataset",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Custom path to output SQLite database. Default: results/blocking/blocking_<split>.db",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Batch size for executemany inserts. Default: 50,000",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit per source file (useful for testing/debugging).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    split_dir = args.data_root / args.split
    s2_path = split_dir / f"{args.split}_source2.tsv"
    s3_path = split_dir / f"{args.split}_source3.tsv"

    db_path = args.db_path or (RESULTS_DIR / f"blocking_{args.split}.db")

    print("=" * 70)
    print("AMAZON BUSINESS ENTITY RESOLUTION")
    print("ENHANCED BLOCKING INDEX BUILDER")
    print("=" * 70)
    print(f"Split      : {args.split}")
    print(f"Source 2   : {s2_path}")
    print(f"Source 3   : {s3_path}")
    print(f"Database   : {db_path}")
    print(f"Batch Size : {args.batch_size:,}")
    if args.limit:
        print(f"Limit      : {args.limit:,} rows per source")

    t_start = time.time()
    connection = init_database(db_path)

    try:
        process_source(connection, "S2", s2_path, batch_size=args.batch_size, limit=args.limit)
        process_source(connection, "S3", s3_path, batch_size=args.batch_size, limit=args.limit)

        create_indexes(connection)
        print_summary(connection, db_path)

    finally:
        connection.close()

    total_time = time.time() - t_start
    print()
    print("=" * 70)
    print(f"BLOCKING INDEX COMPLETE in {total_time:.1f}s ({total_time / 60:.2f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
