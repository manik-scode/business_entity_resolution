from pathlib import Path
import csv
import re
import sys
import unicodedata
from collections import Counter


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_ROOT = (
    PROJECT_ROOT
    / "student_resource"
    / "dataset"
    / "train"
)

S1_FILE = TRAIN_ROOT / "train_source1.tsv"
S2_FILE = TRAIN_ROOT / "train_source2.tsv"
S3_FILE = TRAIN_ROOT / "train_source3.tsv"
GT_FILE = TRAIN_ROOT / "train_ground_truth.tsv"


# =========================================================
# CONFIG
# =========================================================

# We don't need all 2.2M GT rows for this first analysis.
MAX_POSITIVE_PAIRS = 100_000

# Columns we expect in each source file. If a source is missing one,
# we fail fast with a clear message instead of crashing mid-scan.
REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address"]
REQUIRED_GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(value: str) -> str:

    if not value:
        return ""

    value = value.lower().strip()

    # Unicode normalization
    value = unicodedata.normalize("NFKC", value)

    # Replace punctuation with spaces
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Collapse whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


def normalize_name(value: str) -> str:

    value = normalize_text(value)

    # Common legal suffix normalization
    replacements = {
        "corporation": "corp",
        "company": "co",
        "incorporated": "inc",
        "limited": "ltd",
        "private": "pvt",
    }

    words = value.split()

    words = [
        replacements.get(word, word)
        for word in words
    ]

    # Sort only for analysis of word-order robustness.
    # Keep both representations.
    return " ".join(words)


def normalize_name_sorted(value: str) -> str:

    value = normalize_name(value)

    words = value.split()

    return " ".join(sorted(words))


def normalize_address(value: str) -> str:

    if not value:
        return ""

    value = normalize_text(value)

    replacements = {
        "road": "rd",
        "street": "st",
        "avenue": "ave",
        "boulevard": "blvd",
        "drive": "dr",
        "lane": "ln",
        "highway": "hwy",
        "apartment": "apt",
        "suite": "ste",
    }

    words = value.split()

    words = [
        replacements.get(word, word)
        for word in words
    ]

    return " ".join(words)


# =========================================================
# FILE / SCHEMA VALIDATION
# =========================================================

def require_file(path: Path):
    if not path.exists():
        print(f"ERROR: required file not found: {path}", file=sys.stderr)
        sys.exit(1)


def require_columns(path: Path, fieldnames, required_columns):
    missing = [c for c in required_columns if c not in (fieldnames or [])]
    if missing:
        print(
            f"ERROR: {path.name} is missing required column(s) {missing}. "
            f"Found columns: {fieldnames}",
            file=sys.stderr,
        )
        sys.exit(1)


# =========================================================
# READ REQUIRED IDs FROM GROUND TRUTH
# =========================================================

def load_positive_pairs():

    print("Reading ground truth...")

    require_file(GT_FILE)

    positive_pairs = []
    skipped_bad_rows = 0

    with open(
        GT_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t"
        )

        require_columns(GT_FILE, reader.fieldnames, REQUIRED_GT_COLUMNS)

        for row in reader:

            s1_id = row.get("source1_entity_id", "").strip()
            matched = row.get("matched_entity_ids", "").strip()

            if not s1_id or not matched:
                skipped_bad_rows += 1
                continue

            for target_id in matched.split(","):

                target_id = target_id.strip()

                if target_id:
                    positive_pairs.append(
                        (s1_id, target_id)
                    )

                    if len(positive_pairs) >= MAX_POSITIVE_PAIRS:
                        if skipped_bad_rows:
                            print(f"Skipped {skipped_bad_rows:,} row(s) with no S1 id or no matches.")
                        return positive_pairs

    if skipped_bad_rows:
        print(f"Skipped {skipped_bad_rows:,} row(s) with no S1 id or no matches.")

    return positive_pairs


# =========================================================
# COLLECT REQUIRED IDS
# =========================================================

def collect_ids(pairs):

    s1_ids = set()
    s2_ids = set()
    s3_ids = set()
    unknown_prefix = Counter()

    for s1_id, target_id in pairs:

        s1_ids.add(s1_id)

        if target_id.startswith("S2-"):
            s2_ids.add(target_id)

        elif target_id.startswith("S3-"):
            s3_ids.add(target_id)

        else:
            # Don't silently drop unexpected id schemes - surface them.
            prefix = target_id.split("-")[0] if "-" in target_id else target_id
            unknown_prefix[prefix] += 1

    if unknown_prefix:
        print("\nWARNING: some matched_entity_ids had an unrecognized prefix (ignored):")
        for prefix, count in unknown_prefix.most_common():
            print(f"  {prefix}: {count:,}")

    return s1_ids, s2_ids, s3_ids


# =========================================================
# FIND RECORDS
# =========================================================

def load_required_records(path, required_ids):

    records = {}

    if not required_ids:
        return records

    require_file(path)

    print(f"Scanning {path.name}...")

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t"
        )

        require_columns(path, reader.fieldnames, REQUIRED_SOURCE_COLUMNS)

        for row in reader:

            entity_id = row.get("entity_id", "")

            if entity_id in required_ids:

                records[entity_id] = row

                if len(records) == len(required_ids):
                    break

    missing = len(required_ids) - len(records)
    if missing > 0:
        print(f"  Note: {missing:,} required id(s) from {path.name} were not found in the file.")

    return records


# =========================================================
# ANALYZE
# =========================================================

def analyze():

    pairs = load_positive_pairs()

    print()
    print("=" * 70)
    print("POSITIVE PAIRS")
    print("=" * 70)

    print(f"Positive pairs loaded: {len(pairs):,}")

    if not pairs:
        print("No positive pairs loaded - nothing to analyze.")
        return

    s1_ids, s2_ids, s3_ids = collect_ids(pairs)

    print(f"S1 records required: {len(s1_ids):,}")
    print(f"S2 records required: {len(s2_ids):,}")
    print(f"S3 records required: {len(s3_ids):,}")

    print()
    print("Loading matching records...")

    s1_records = load_required_records(
        S1_FILE,
        s1_ids
    )

    s2_records = load_required_records(
        S2_FILE,
        s2_ids
    )

    s3_records = load_required_records(
        S3_FILE,
        s3_ids
    )

    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    exact_name = 0
    sorted_name = 0
    exact_address = 0
    both_name_address = 0

    by_source = Counter()
    missing_records = 0

    for s1_id, target_id in pairs:

        s1 = s1_records.get(s1_id)

        if target_id.startswith("S2-"):
            target = s2_records.get(target_id)
            source = "S2"

        elif target_id.startswith("S3-"):
            target = s3_records.get(target_id)
            source = "S3"

        else:
            continue  # unrecognized prefix, already warned about above

        if not s1 or not target:
            missing_records += 1
            continue

        by_source[source] += 1

        s1_name = normalize_name(
            s1.get("business_name", "")
        )

        target_name = normalize_name(
            target.get("business_name", "")
        )

        s1_sorted = normalize_name_sorted(
            s1.get("business_name", "")
        )

        target_sorted = normalize_name_sorted(
            target.get("business_name", "")
        )

        s1_address = normalize_address(
            s1.get("business_address", "")
        )

        target_address = normalize_address(
            target.get("business_address", "")
        )

        name_match = (
            s1_name != ""
            and s1_name == target_name
        )

        sorted_match = (
            s1_sorted != ""
            and s1_sorted == target_sorted
        )

        address_match = (
            s1_address != ""
            and s1_address == target_address
        )

        if name_match:
            exact_name += 1

        if sorted_match:
            sorted_name += 1

        if address_match:
            exact_address += 1

        if name_match and address_match:
            both_name_address += 1

    analyzed = sum(by_source.values())

    print()
    print(f"Positive pairs loaded : {len(pairs):,}")
    print(f"Pairs analyzed         : {analyzed:,}")
    print(f"Pairs skipped (missing record): {missing_records:,}")

    if analyzed == 0:
        print("\nNo pairs had both records available - cannot compute blocking signals.")
        return

    print()
    print("Source distribution (of analyzed pairs):")

    for source, count in by_source.most_common():
        print(f"{source}: {count:,}")

    print()
    print("BLOCKING SIGNALS")
    print("(percentages are of ANALYZED pairs, not all loaded pairs)")
    print("-" * 70)

    print(
        f"Exact normalized name       : "
        f"{exact_name:,} "
        f"({exact_name / analyzed * 100:.2f}%)"
    )

    print(
        f"Word-order normalized name  : "
        f"{sorted_name:,} "
        f"({sorted_name / analyzed * 100:.2f}%)"
    )

    print(
        f"Exact normalized address    : "
        f"{exact_address:,} "
        f"({exact_address / analyzed * 100:.2f}%)"
    )

    print(
        f"Name + address exact        : "
        f"{both_name_address:,} "
        f"({both_name_address / analyzed * 100:.2f}%)"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    analyze()