import argparse
import csv
from collections import Counter, defaultdict
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
TRAIN_ROOT = DATA_ROOT / "train"

DB_PATH = PROJECT_ROOT / "results" / "blocking" / "blocking_train.db"
S1_PATH = TRAIN_ROOT / "train_source1.tsv"
GROUND_TRUTH_PATH = TRAIN_ROOT / "train_ground_truth.tsv"


# =========================================================
# NORMALIZATION & KEY EXTRACTION
# Must stay strictly aligned with build_index.py
# =========================================================

PUNCT_SYMBOL_TABLE = {
    codepoint: " "
    for codepoint in range(0x10000)
    if unicodedata.category(chr(codepoint))[0] in ("P", "S")
}

NUM_RE = re.compile(r"\b\d+[a-z]?\b")

NAME_REPLACEMENTS = {
    "corporation": "corp",
    "company": "co",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "public": "pub",
    "societe": "soc",
    "etablissement": "etb",
    "प्राइवेट": "pvt",
    "लिमिटेड": "ltd",
    "कंपनी": "co",
    "प्रा": "pvt",
    "लि": "ltd",
}

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


def normalize_text(value: str) -> str:
    """Normalize unicode, lowercase, and replace punctuation & symbols with spaces."""
    if not value:
        return ""
    return unicodedata.normalize("NFKC", value).lower().translate(PUNCT_SYMBOL_TABLE)


def extract_name_keys(raw_name: str):
    """
    Single-pass extraction of all name-based blocking keys.
    Returns:
        (name_key, sorted_name_key, name_token_key, first_name_token)
    """
    if not raw_name:
        return "", "", "", ""

    cleaned = normalize_text(raw_name)
    raw_tokens = cleaned.split()
    if not raw_tokens:
        return "", "", "", ""

    tokens = [NAME_REPLACEMENTS.get(tok, tok) for tok in raw_tokens]

    name_key = " ".join(tokens)
    sorted_name_key = " ".join(sorted(tokens))

    candidate_tokens = [tok for tok in tokens if tok not in ALL_STOPWORDS and len(tok) >= 3]

    if candidate_tokens:
        name_token_key = max(candidate_tokens, key=len)
    else:
        non_legal = [tok for tok in tokens if tok not in IGNORED_LEGAL]
        name_token_key = max(non_legal, key=len) if non_legal else tokens[0]

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

    num_match = NUM_RE.search(cleaned)
    address_number = num_match.group(0) if num_match else ""

    norm_tokens = [ADDRESS_REPLACEMENTS.get(tok, tok) for tok in tokens]
    address_key = " ".join(norm_tokens)

    return address_number, address_key


# =========================================================
# LOAD GROUND TRUTH
# =========================================================

def load_ground_truth(gt_path: Path, limit=None):
    print()
    print("=" * 70)
    print("LOADING GROUND TRUTH")
    print("=" * 70)

    ground_truth = {}
    with open(gt_path, "r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"]
            matched_raw = (row.get("matched_entity_ids", "") or "").strip()

            if matched_raw:
                matches = set(x.strip() for x in matched_raw.split(",") if x.strip())
            else:
                matches = set()

            ground_truth[s1_id] = matches
            if limit and len(ground_truth) >= limit:
                break

    print(f"Loaded ground truth for {len(ground_truth):,} S1 entities")
    return ground_truth


# =========================================================
# BUILD CANDIDATES FOR ONE S1 (WITH RULE ATTRIBUTION)
# =========================================================

def get_candidates_by_rule(
    cursor,
    country,
    name_key,
    sorted_name_key,
    name_token_key,
    first_name_token,
    address_number,
    address_key,
):
    """
    Retrieves candidate entity_ids per blocking rule to evaluate
    both union recall and individual rule contributions.
    """
    rule_candidates = {}

    # Block 1: Exact normalized name
    if name_key:
        cursor.execute(
            "SELECT entity_id FROM records WHERE country = ? AND name_key = ?",
            (country, name_key),
        )
        rule_candidates[1] = set(row[0] for row in cursor.fetchall())
    else:
        rule_candidates[1] = set()

    # Block 2: Word-order invariant name
    if sorted_name_key:
        cursor.execute(
            "SELECT entity_id FROM records WHERE country = ? AND sorted_name_key = ?",
            (country, sorted_name_key),
        )
        rule_candidates[2] = set(row[0] for row in cursor.fetchall())
    else:
        rule_candidates[2] = set()

    # Block 3: Distinctive token + address number
    if name_token_key and address_number:
        cursor.execute(
            "SELECT entity_id FROM records WHERE country = ? AND name_token_key = ? AND address_number = ?",
            (country, name_token_key, address_number),
        )
        rule_candidates[3] = set(row[0] for row in cursor.fetchall())
    else:
        rule_candidates[3] = set()

    # Block 4: First token + address number
    if first_name_token and address_number:
        cursor.execute(
            "SELECT entity_id FROM records WHERE country = ? AND first_name_token = ? AND address_number = ?",
            (country, first_name_token, address_number),
        )
        rule_candidates[4] = set(row[0] for row in cursor.fetchall())
    else:
        rule_candidates[4] = set()

    # Block 5: Exact normalized address
    if address_key:
        cursor.execute(
            "SELECT entity_id FROM records WHERE country = ? AND address_key = ?",
            (country, address_key),
        )
        rule_candidates[5] = set(row[0] for row in cursor.fetchall())
    else:
        rule_candidates[5] = set()

    all_candidates = (
        rule_candidates[1]
        | rule_candidates[2]
        | rule_candidates[3]
        | rule_candidates[4]
        | rule_candidates[5]
    )

    return all_candidates, rule_candidates


# =========================================================
# MAIN EVALUATION
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SQLite Blocking Recall for Entity Resolution"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100_000,
        help="Number of S1 entities to evaluate (default: 100,000, set 0 for all).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help=f"Path to SQLite blocking database (default: {DB_PATH})",
    )
    parser.add_argument(
        "--s1-path",
        type=Path,
        default=S1_PATH,
        help=f"Path to Source 1 TSV (default: {S1_PATH})",
    )
    parser.add_argument(
        "--gt-path",
        type=Path,
        default=GROUND_TRUTH_PATH,
        help=f"Path to Ground Truth TSV (default: {GROUND_TRUTH_PATH})",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10_000,
        help="Print progress every N records (default: 10,000)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eval_limit = args.limit if args.limit > 0 else None

    print("=" * 70)
    print("AMAZON BUSINESS ENTITY RESOLUTION")
    print("BLOCKING RECALL EVALUATION (ENHANCED)")
    print("=" * 70)
    print(f"Database         : {args.db_path}")
    print(f"Evaluation limit : {eval_limit:,}" if eval_limit else "Evaluation limit : ALL")

    if not args.db_path.exists():
        raise FileNotFoundError(f"Blocking database not found:\n{args.db_path}")

    ground_truth = load_ground_truth(args.gt_path, eval_limit)

    connection = sqlite3.connect(args.db_path)
    # Read-only performance PRAGMAs to maximize throughput
    connection.execute("PRAGMA query_only = ON;")
    connection.execute("PRAGMA cache_size = -131072;")  # 128MB cache
    connection.execute("PRAGMA mmap_size = 30000000000;")  # 30GB mmap
    cursor = connection.cursor()

    # Get total indexed records in DB for Reduction Ratio calculation
    cursor.execute("SELECT COUNT(*) FROM records")
    total_indexed_records = cursor.fetchone()[0]

    total_s1 = 0
    total_true_matches = 0
    total_candidates = 0
    true_matches_found = 0
    zero_candidate_count = 0

    candidate_size_counts = Counter()

    recall_by_match_count = defaultdict(
        lambda: {"entities": 0, "true_matches": 0, "found_matches": 0}
    )
    block_hits = Counter()

    # Per-rule attribution tracking
    rule_total_candidates = Counter()
    rule_total_hits = Counter()
    rule_unique_hits = Counter()

    start_time = time.time()

    print()
    print("=" * 70)
    print("EVALUATING S1")
    print("=" * 70)

    with open(args.s1_path, "r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            s1_id = row["entity_id"]
            if s1_id not in ground_truth:
                continue

            business_name = (row.get("business_name", "") or "")
            business_address = (row.get("business_address", "") or "")
            country = (row.get("country", "") or "").strip().upper()

            name_key, sorted_name_key, name_token_key, first_name_token = extract_name_keys(
                business_name
            )
            address_number, address_key = extract_address_keys(business_address)

            candidates, rule_cands = get_candidates_by_rule(
                cursor=cursor,
                country=country,
                name_key=name_key,
                sorted_name_key=sorted_name_key,
                name_token_key=name_token_key,
                first_name_token=first_name_token,
                address_number=address_number,
                address_key=address_key,
            )

            true_matches = ground_truth[s1_id]
            candidate_count = len(candidates)

            candidate_size_counts[candidate_count] += 1
            total_candidates += candidate_count
            total_s1 += 1
            total_true_matches += len(true_matches)

            if candidate_count == 0:
                zero_candidate_count += 1

            found = true_matches & candidates
            true_matches_found += len(found)

            # Per-rule attribution for true matches
            for r_idx, cands in rule_cands.items():
                rule_total_candidates[r_idx] += len(cands)

            for tm in true_matches:
                matching_rules = [r for r, cands in rule_cands.items() if tm in cands]
                for r in matching_rules:
                    rule_total_hits[r] += 1
                if len(matching_rules) == 1:
                    rule_unique_hits[matching_rules[0]] += 1

            match_count = len(true_matches)
            stats = recall_by_match_count[match_count]
            stats["entities"] += 1
            stats["true_matches"] += len(true_matches)
            stats["found_matches"] += len(found)

            if true_matches:
                if true_matches <= candidates:
                    block_hits["perfect_entity_recall"] += 1
                if found:
                    block_hits["at_least_one_match"] += 1
            else:
                if not candidates:
                    block_hits["correct_zero_candidates"] += 1

            if total_s1 % args.progress_interval == 0:
                elapsed = time.time() - start_time
                rate = total_s1 / elapsed if elapsed > 0 else 0
                current_recall = (
                    true_matches_found / total_true_matches * 100
                    if total_true_matches
                    else 0
                )
                print(
                    f"Processed {total_s1:,} S1 | {rate:,.0f} S1/s | "
                    f"Recall: {current_recall:.2f}% | Candidates: {total_candidates:,}"
                )

            if eval_limit and total_s1 >= eval_limit:
                break

    connection.close()

    # =====================================================
    # METRICS CALCULATION
    # =====================================================

    elapsed = time.time() - start_time
    recall = true_matches_found / total_true_matches if total_true_matches else 0
    avg_candidates = total_candidates / total_s1 if total_s1 else 0

    # Percentiles from counter without sorting huge array
    def get_percentile(size_counts, total_items, p):
        target = int(total_items * p)
        cumulative = 0
        for size in sorted(size_counts):
            cumulative += size_counts[size]
            if cumulative >= target:
                return size
        return 0

    p50 = get_percentile(candidate_size_counts, total_s1, 0.50)
    p90 = get_percentile(candidate_size_counts, total_s1, 0.90)
    p95 = get_percentile(candidate_size_counts, total_s1, 0.95)
    p99 = get_percentile(candidate_size_counts, total_s1, 0.99)
    max_candidates = max(candidate_size_counts.keys()) if candidate_size_counts else 0

    # Standard Entity Resolution Metrics
    # Reduction Ratio = 1 - (Total Candidates / (|S1| * (|S2| + |S3|)))
    cartesian_space = total_s1 * total_indexed_records
    reduction_ratio = (
        (1.0 - (total_candidates / cartesian_space)) * 100
        if cartesian_space
        else 100.0
    )
    pair_quality = (
        (true_matches_found / total_candidates) * 100
        if total_candidates
        else 0.0
    )

    # =====================================================
    # REPORT
    # =====================================================

    print()
    print("=" * 70)
    print("BLOCKING EVALUATION RESULTS")
    print("=" * 70)

    print(f"S1 entities evaluated       : {total_s1:,}")
    print(f"Total indexed in database   : {total_indexed_records:,} (S2 + S3)")
    print(f"True matches                : {total_true_matches:,}")
    print(f"True matches retrieved      : {true_matches_found:,}")

    print()
    print(f"BLOCKING RECALL             : {recall * 100:.4f}%")
    print(f"Pairs Quality (Precision)   : {pair_quality:.4f}%")
    print(f"Reduction Ratio             : {reduction_ratio:.6f}%")

    print()
    print(f"Average candidates / S1     : {avg_candidates:.2f}")
    print(f"Median candidates (P50)     : {p50:,}")
    print(f"P90 candidates              : {p90:,}")
    print(f"P95 candidates              : {p95:,}")
    print(f"P99 candidates              : {p99:,}")
    print(f"Maximum candidates          : {max_candidates:,}")

    print()
    print(f"S1 with zero candidates     : {zero_candidate_count:,}")
    if total_s1:
        print(f"Zero-candidate rate         : {zero_candidate_count / total_s1 * 100:.4f}%")

    # =====================================================
    # PER-RULE PERFORMANCE & ATTRIBUTION
    # =====================================================

    rule_descriptions = {
        1: "Exact normalized name",
        2: "Word-order invariant name",
        3: "Name token + address number",
        4: "First name token + address number",
        5: "Exact normalized address",
    }

    print()
    print("=" * 70)
    print("PER-RULE PERFORMANCE & ATTRIBUTION")
    print("=" * 70)
    print(
        f"{'Rule':<4} {'Description':<35} {'Total Hits':>11} {'Unique Hits':>12} {'Candidates':>12}"
    )
    print("-" * 78)

    for r_idx in range(1, 6):
        hits = rule_total_hits[r_idx]
        u_hits = rule_unique_hits[r_idx]
        cands = rule_total_candidates[r_idx]
        desc = rule_descriptions[r_idx]
        print(f"{r_idx:<4} {desc:<35} {hits:>11,} {u_hits:>12,} {cands:>12,}")

    # =====================================================
    # ENTITY-LEVEL RECALL
    # =====================================================

    print()
    print("=" * 70)
    print("ENTITY-LEVEL RECALL")
    print("=" * 70)

    print(
        f"{'Matches':>10} {'Entities':>12} {'True':>12} {'Found':>12} {'Recall':>12}"
    )
    print("-" * 70)

    for match_count in sorted(recall_by_match_count):
        stats = recall_by_match_count[match_count]
        entity_recall = (
            stats["found_matches"] / stats["true_matches"]
            if stats["true_matches"]
            else 0
        )
        print(
            f"{match_count:>10} "
            f"{stats['entities']:>12,} "
            f"{stats['true_matches']:>12,} "
            f"{stats['found_matches']:>12,} "
            f"{entity_recall * 100:>11.2f}%"
        )

    # =====================================================
    # ADDITIONAL STATISTICS
    # =====================================================

    print()
    print("=" * 70)
    print("ADDITIONAL STATISTICS")
    print("=" * 70)

    for key, value in block_hits.items():
        print(f"{key:<35}: {value:,}")

    print()
    print(f"Evaluation time: {elapsed:.1f}s ({elapsed / 60:.2f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
