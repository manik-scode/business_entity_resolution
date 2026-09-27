import csv
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "results" / "blocking" / "blocking_train_v2.db"
GT_PATH = ROOT / "student_resource" / "dataset" / "train" / "train_ground_truth.tsv"

SAMPLE_SIZE = 1000


# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------

REPLACEMENTS = {
    "corporation": "corp",
    "company": "co",
    "incorporated": "inc",
    "limited": "ltd",
    "private": "pvt",
    "public": "pub",
    "societe": "soc",
    "etablissement": "etb",
}

GENERIC_NAME = {
    "corp", "co", "inc", "ltd", "pvt", "pub",
    "llc", "llp", "plc", "sa", "sarl",
    "gmbh", "ag", "limited", "company", "corporation"
}

GENERIC_ADDRESS = {
    "road", "rd", "street", "st", "lane", "ln",
    "avenue", "ave", "boulevard", "blvd",
    "highway", "hwy", "sector", "sec",
    "block", "district", "dist", "city",
    "town", "village", "near", "opposite",
    "opp", "india", "usa", "us", "france"
}


def normalize_text(text):

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()

    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)

    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def name_tokens(text):

    text = normalize_text(text)

    return [
        x for x in text.split()
        if x and x not in GENERIC_NAME
    ]


def address_tokens(text):

    text = normalize_text(text)

    return [
        x for x in text.split()
        if x and x not in GENERIC_ADDRESS
    ]


def address_numbers(text):

    text = normalize_text(text)

    return set(
        re.findall(r"\b\d+[a-z]?\b", text)
    )


def jaccard(a, b):

    a = set(a)
    b = set(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def longest_common_prefix(a, b):

    n = min(len(a), len(b))

    i = 0

    while i < n and a[i] == b[i]:
        i += 1

    return i


def longest_common_suffix(a, b):

    a = a[::-1]
    b = b[::-1]

    return longest_common_prefix(a, b)


# ------------------------------------------------------------
# LOAD SAMPLE + GROUND TRUTH
# ------------------------------------------------------------

def main():

    print("=" * 78)
    print("AMAZON ML CHALLENGE — V7 BLOCKING FORENSIC ANALYSIS")
    print("=" * 78)

    conn = sqlite3.connect(str(DB_PATH))

    conn.execute("PRAGMA cache_size=-300000")
    conn.execute("PRAGMA temp_store=MEMORY")

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    conn.execute("DROP TABLE IF EXISTS temp.v7_s1")

    conn.execute(f"""
        CREATE TEMP TABLE v7_s1 AS
        SELECT *
        FROM s1
        ORDER BY rowid
        LIMIT {SAMPLE_SIZE}
    """)

    sample_ids = {
        row[0]
        for row in conn.execute(
            "SELECT entity_id FROM v7_s1"
        )
    }

    print(f"\nSample S1 rows: {len(sample_ids):,}")

    # --------------------------------------------------------
    # LOAD RAW S1 DATA
    # --------------------------------------------------------

    s1_raw = {}

    train_s1_path = (
        ROOT
        / "student_resource"
        / "dataset"
        / "train"
        / "train_source1.tsv"
    )

    with open(
        train_s1_path,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            sid = row["entity_id"]

            if sid in sample_ids:

                s1_raw[sid] = {
                    "name": row["business_name"] or "",
                    "address": row["business_address"] or "",
                    "country": row["country"] or "",
                }

            if len(s1_raw) == len(sample_ids):
                break

    print(f"S1 raw rows loaded: {len(s1_raw):,}")

    # --------------------------------------------------------
    # LOAD GROUND TRUTH
    # --------------------------------------------------------

    gt = {}

    with open(
        GT_PATH,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            sid = row["source1_entity_id"]

            if sid not in sample_ids:
                continue

            raw = (row["matched_entity_ids"] or "").strip()

            gt[sid] = (
                {
                    x.strip()
                    for x in raw.split(",")
                    if x.strip()
                }
                if raw
                else set()
            )

            if len(gt) == len(sample_ids):
                break

    print(f"Ground truth rows loaded: {len(gt):,}")

    # --------------------------------------------------------
    # RECREATE V5/V6 CANDIDATES
    # --------------------------------------------------------

    conn.execute("DROP TABLE IF EXISTS temp.v7_candidates")

    conn.execute("""
        CREATE TEMP TABLE v7_candidates (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,
            PRIMARY KEY (
                source1_entity_id,
                candidate_entity_id
            )
        )
    """)

    # --------------------------------------------------------
    # BASE BLOCKING
    # --------------------------------------------------------

    rules = [

        """
        INSERT OR IGNORE INTO v7_candidates
        SELECT s.entity_id, r.entity_id
        FROM v7_s1 s
        JOIN records r
          ON r.country=s.country
         AND r.name_key=s.name_key
        WHERE r.source IN ('S2','S3')
        """,

        """
        INSERT OR IGNORE INTO v7_candidates
        SELECT s.entity_id, r.entity_id
        FROM v7_s1 s
        JOIN records r
          ON r.country=s.country
         AND r.sorted_name_key=s.sorted_name_key
        WHERE r.source IN ('S2','S3')
        """,

        """
        INSERT OR IGNORE INTO v7_candidates
        SELECT s.entity_id, r.entity_id
        FROM v7_s1 s
        JOIN records r
          ON r.country=s.country
         AND r.name_token_key=s.name_token_key
         AND r.address_number=s.address_number
        WHERE r.source IN ('S2','S3')
        """,

        """
        INSERT OR IGNORE INTO v7_candidates
        SELECT s.entity_id, r.entity_id
        FROM v7_s1 s
        JOIN records r
          ON r.country=s.country
         AND r.first_name_token=s.first_name_token
         AND r.address_number=s.address_number
        WHERE r.source IN ('S2','S3')
        """,

        """
        INSERT OR IGNORE INTO v7_candidates
        SELECT s.entity_id, r.entity_id
        FROM v7_s1 s
        JOIN records r
          ON r.country=s.country
         AND r.first_name_token=s.first_name_token
         AND r.address_number=s.address_number
         AND r.name_prefix_sig=s.name_prefix_sig
        WHERE r.source IN ('S2','S3')
        """,

        """
        INSERT OR IGNORE INTO v7_candidates
        SELECT s.entity_id, r.entity_id
        FROM v7_s1 s
        JOIN records r
          ON r.country=s.country
         AND r.first_name_token=s.first_name_token
         AND r.address_number=s.address_number
         AND r.name_suffix_sig=s.name_suffix_sig
        WHERE r.source IN ('S2','S3')
        """,

        """
        INSERT OR IGNORE INTO v7_candidates
        SELECT s.entity_id, r.entity_id
        FROM v7_s1 s
        JOIN records r
          ON r.country=s.country
         AND r.address_key=s.address_key
         AND r.first_name_token=s.first_name_token
        WHERE r.source IN ('S2','S3')
        """,

        """
        INSERT OR IGNORE INTO v7_candidates
        SELECT s.entity_id, r.entity_id
        FROM v7_s1 s
        JOIN records r
          ON r.country=s.country
         AND r.name_prefix_sig=s.name_prefix_sig
         AND r.address_token_sig=s.address_token_sig
        WHERE r.source IN ('S2','S3')
        """,

        """
        INSERT OR IGNORE INTO v7_candidates
        SELECT s.entity_id, r.entity_id
        FROM v7_s1 s
        JOIN records r
          ON r.country=s.country
         AND r.name_suffix_sig=s.name_suffix_sig
         AND r.address_token_sig=s.address_token_sig
        WHERE r.source IN ('S2','S3')
        """,

        """
        INSERT OR IGNORE INTO v7_candidates
        SELECT s.entity_id, r.entity_id
        FROM v7_s1 s
        JOIN records r
          ON r.country=s.country
         AND r.first_name_token=s.first_name_token
         AND r.address_token_sig=s.address_token_sig
        WHERE r.source IN ('S2','S3')
        """,

    ]

    print("\nBuilding V5/V6 candidate set...")

    for i, sql in enumerate(rules, 1):

        conn.execute(sql)

        print(
            f"Rule {i:02d} complete | "
            f"candidates = "
            f"{conn.execute('SELECT COUNT(*) FROM v7_candidates').fetchone()[0]:,}"
        )

    conn.commit()

    # --------------------------------------------------------
    # FIND MISSED TRUE PAIRS
    # --------------------------------------------------------

    print("\nFinding missed true matches...")

    candidates = {}

    for sid, cid in conn.execute("""
        SELECT source1_entity_id,
               candidate_entity_id
        FROM v7_candidates
    """):

        candidates.setdefault(
            sid,
            set()
        ).add(cid)

    missed = []

    for sid in sample_ids:

        truth = gt.get(sid, set())
        cand = candidates.get(sid, set())

        for cid in truth - cand:

            missed.append(
                (sid, cid)
            )

    print(
        f"\nTrue pairs: "
        f"{sum(len(x) for x in gt.values()):,}"
    )

    print(
        f"Missed true pairs: "
        f"{len(missed):,}"
    )

    # --------------------------------------------------------
    # FETCH ACTUAL RECORDS FOR MISSED PAIRS
    # --------------------------------------------------------

    s1_rows = {}

    for sid in sample_ids:

        row = conn.execute("""
            SELECT
                entity_id,
                country,
                name_key,
                sorted_name_key,
                name_token_key,
                first_name_token,
                address_number,
                address_key,
                name_prefix_sig,
                name_suffix_sig,
                address_token_sig
            FROM s1
            WHERE entity_id=?
        """, (sid,)).fetchone()

        if row:
            s1_rows[sid] = row

    record_ids = [
        cid
        for _, cid in missed
    ]

    records = {}

    # SQLite parameter limit workaround
    for start in range(0, len(record_ids), 500):

        chunk = record_ids[start:start + 500]

        placeholders = ",".join(
            "?" for _ in chunk
        )

        rows = conn.execute(
            f"""
            SELECT
                entity_id,
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
            """,
            chunk
        ).fetchall()

        for row in rows:
            records[row[0]] = row

    # --------------------------------------------------------
    # ANALYZE MISSES
    # --------------------------------------------------------

    categories = Counter()

    name_exact = 0
    sorted_exact = 0
    token_exact = 0
    first_token_exact = 0
    number_exact = 0
    address_exact = 0
    address_token_exact = 0
    prefix_exact = 0
    suffix_exact = 0

    examples = []

    for sid, cid in missed:

        s = s1_raw.get(sid)

        r = records.get(cid)

        if not s or not r:
            continue

        s_name = normalize_text(s["name"])
        r_name = normalize_text(r[2])

        s_nt = name_tokens(s["name"])
        r_nt = name_tokens(r[2])

        s_at = address_tokens(s["address"])
        r_at = address_tokens(r[3])

        s_nums = address_numbers(s["address"])
        r_nums = address_numbers(r[3])

        if s_name == r_name and s_name:
            name_exact += 1

        if sorted(s_nt) == sorted(r_nt) and s_nt:
            sorted_exact += 1

        if set(s_nt) == set(r_nt) and s_nt:
            token_exact += 1

        if (
            s_nt
            and r_nt
            and s_nt[0] == r_nt[0]
        ):
            first_token_exact += 1

        if s_nums & r_nums:
            number_exact += 1

        s_addr = normalize_text(s["address"])
        r_addr = normalize_text(r[3])

        if s_addr == r_addr and s_addr:
            address_exact += 1

        if set(s_at) & set(r_at):
            address_token_exact += 1

        if (
            s_name[:4]
            and s_name[:4] == r_name[:4]
        ):
            prefix_exact += 1

        if (
            s_name[-4:]
            and s_name[-4:] == r_name[-4:]
        ):
            suffix_exact += 1

        # ----------------------------------------------------
        # CATEGORIZE
        # ----------------------------------------------------

        if not s["address"] or not r[3]:
            categories["missing_address"] += 1

        elif not s_name or not r_name:
            categories["missing_name"] += 1

        elif s_nums and r_nums and not (s_nums & r_nums):
            categories["address_number_mismatch"] += 1

        elif (
            set(s_nt) & set(r_nt)
            and set(s_at) & set(r_at)
        ):
            categories["partial_name_and_address_overlap"] += 1

        elif set(s_nt) & set(r_nt):
            categories["partial_name_overlap"] += 1

        elif set(s_at) & set(r_at):
            categories["partial_address_overlap"] += 1

        else:
            categories["heavy_name_address_noise"] += 1

        # Save first few examples
        if len(examples) < 20:

            examples.append({
                "sid": sid,
                "cid": cid,
                "s1_name": s["name"],
                "s2_name": r[2],
                "s1_address": s["address"],
                "s2_address": r[3],
            })

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("V7 MISSED-PAIR FORENSICS")
    print("=" * 78)

    print(
        f"\nMissed true pairs              : "
        f"{len(missed):,}"
    )

    print("\nFeature overlap among MISSED pairs:")

    print(
        f"Name exact after normalization : "
        f"{name_exact:,} "
        f"({name_exact/len(missed)*100:.2f}%)"
    )

    print(
        f"Sorted name exact               : "
        f"{sorted_exact:,} "
        f"({sorted_exact/len(missed)*100:.2f}%)"
    )

    print(
        f"Name token set exact            : "
        f"{token_exact:,} "
        f"({token_exact/len(missed)*100:.2f}%)"
    )

    print(
        f"First name token same           : "
        f"{first_token_exact:,} "
        f"({first_token_exact/len(missed)*100:.2f}%)"
    )

    print(
        f"Address number overlap          : "
        f"{number_exact:,} "
        f"({number_exact/len(missed)*100:.2f}%)"
    )

    print(
        f"Address exact after normalization: "
        f"{address_exact:,} "
        f"({address_exact/len(missed)*100:.2f}%)"
    )

    print(
        f"Address token overlap            : "
        f"{address_token_exact:,} "
        f"({address_token_exact/len(missed)*100:.2f}%)"
    )

    print(
        f"Name prefix overlap              : "
        f"{prefix_exact:,} "
        f"({prefix_exact/len(missed)*100:.2f}%)"
    )

    print(
        f"Name suffix overlap              : "
        f"{suffix_exact:,} "
        f"({suffix_exact/len(missed)*100:.2f}%)"
    )

    print("\nMiss categories:")

    for category, count in categories.most_common():

        print(
            f"{category:<40}"
            f"{count:>8,} "
            f"({count/len(missed)*100:.2f}%)"
        )

    print("\n" + "=" * 78)
    print("EXAMPLE MISSED TRUE PAIRS")
    print("=" * 78)

    for i, ex in enumerate(examples, 1):

        print(f"\n[{i}]")
        print(f"S1 ID      : {ex['sid']}")
        print(f"TRUE ID    : {ex['cid']}")
        print(f"S1 NAME    : {ex['s1_name']}")
        print(f"TRUE NAME  : {ex['s2_name']}")
        print(f"S1 ADDRESS : {ex['s1_address']}")
        print(f"TRUE ADDR  : {ex['s2_address']}")

    print("\n" + "=" * 78)
    print("V7 analysis complete.")
    print("No permanent database tables were modified.")
    print("=" * 78)

    conn.close()


if __name__ == "__main__":
    main()
