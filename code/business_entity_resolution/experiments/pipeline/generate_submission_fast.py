import argparse
import csv
import re
import sqlite3
import time
import unicodedata


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
        "name_token_key": "|".join(sorted_tokens),
        "first_name_token": tokens[0] if tokens else "",
        "name_prefix_sig": (
            f"{compact[:4]}:{(len(compact) // 5) * 5}"
            if compact else ""
        ),
        "name_suffix_sig": (
            f"{compact[-4:]}:{(len(compact) // 5) * 5}"
            if compact else ""
        ),
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
    }


def make_features(row):
    country_raw = (row.get("country") or "").strip()

    country = COUNTRY_MAP.get(
        country_raw.upper(),
        country_raw
    )

    n = name_features(
        row.get("business_name", "")
    )

    a = address_features(
        row.get("business_address", "")
    )

    return (
        row["entity_id"],
        country,
        n["name_key"],
        n["sorted_name_key"],
        n["name_token_key"],
        n["first_name_token"],
        n["name_prefix_sig"],
        n["name_suffix_sig"],
        a["address_key"],
        a["address_number"],
        a["address_token_sig"],
    )


def create_s1_table(conn):
    conn.execute("""
        DROP TABLE IF EXISTS s1
    """)

    conn.execute("""
        CREATE TABLE s1 (
            entity_id TEXT PRIMARY KEY,
            country TEXT,
            name_key TEXT,
            sorted_name_key TEXT,
            name_token_key TEXT,
            first_name_token TEXT,
            name_prefix_sig TEXT,
            name_suffix_sig TEXT,
            address_key TEXT,
            address_number TEXT,
            address_token_sig TEXT
        )
    """)

    conn.execute("""
        CREATE INDEX idx_s1_name
        ON s1(country, name_key)
    """)

    conn.execute("""
        CREATE INDEX idx_s1_sorted_name
        ON s1(country, sorted_name_key)
    """)

    conn.execute("""
        CREATE INDEX idx_s1_token_number
        ON s1(country, name_token_key, address_number)
    """)

    conn.execute("""
        CREATE INDEX idx_s1_first_number
        ON s1(country, first_name_token, address_number)
    """)

    conn.execute("""
        CREATE INDEX idx_s1_prefix_number
        ON s1(country, name_prefix_sig, address_number)
    """)

    conn.execute("""
        CREATE INDEX idx_s1_suffix_number
        ON s1(country, name_suffix_sig, address_number)
    """)

    conn.execute("""
        CREATE INDEX idx_s1_prefix_address
        ON s1(country, name_prefix_sig, address_token_sig)
    """)

    conn.execute("""
        CREATE INDEX idx_s1_suffix_address
        ON s1(country, name_suffix_sig, address_token_sig)
    """)

    conn.execute("""
        CREATE INDEX idx_s1_first_address
        ON s1(country, first_name_token, address_token_sig)
    """)


def load_s1(conn, source1, limit):
    print("Loading S1 features...")

    start = time.time()
    batch = []

    processed = 0

    with open(
        source1,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            if limit and processed >= limit:
                break

            batch.append(
                make_features(row)
            )

            processed += 1

            if len(batch) >= 10000:

                conn.executemany("""
                    INSERT INTO s1 VALUES (
                        ?,?,?,?,?,?,?,?,?,?,?
                    )
                """, batch)

                batch.clear()

                if processed % 100000 == 0:
                    print(
                        f"S1 loaded: {processed:,}"
                    )

        if batch:
            conn.executemany("""
                INSERT INTO s1 VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?
                )
            """, batch)

    conn.commit()

    print(
        f"S1 loaded: {processed:,} "
        f"in {(time.time() - start) / 60:.2f} min"
    )

    return processed


def create_candidate_table(conn):
    conn.execute("""
        DROP TABLE IF EXISTS candidates
    """)

    conn.execute("""
        CREATE TABLE candidates (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,
            PRIMARY KEY (
                source1_entity_id,
                candidate_entity_id
            )
        )
    """)

    conn.execute("""
        CREATE INDEX idx_candidates_s1
        ON candidates(source1_entity_id)
    """)

    conn.commit()


def run_rule(conn, name, sql):
    print(f"Running {name}...")

    start = time.time()

    conn.execute(sql)

    conn.commit()

    count = conn.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    print(
        f"{name} complete | "
        f"Candidates: {count:,} | "
        f"Time: {(time.time() - start) / 60:.2f} min"
    )


def generate_candidates(conn):
    # ---------------------------------------------------------
    # R1 — exact normalized name
    # ---------------------------------------------------------

    run_rule(
        conn,
        "R1 exact name",
        """
        INSERT OR IGNORE INTO candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM s1 s
        JOIN records r
          ON r.country = s.country
         AND r.name_key = s.name_key
        """
    )

    # ---------------------------------------------------------
    # R2 — sorted name
    # ---------------------------------------------------------

    run_rule(
        conn,
        "R2 sorted name",
        """
        INSERT OR IGNORE INTO candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM s1 s
        JOIN records r
          ON r.country = s.country
         AND r.sorted_name_key = s.sorted_name_key
        """
    )

    # ---------------------------------------------------------
    # R3 — name token + address number
    # ---------------------------------------------------------

    run_rule(
        conn,
        "R3 name token + number",
        """
        INSERT OR IGNORE INTO candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM s1 s
        JOIN records r
          ON r.country = s.country
         AND r.name_token_key = s.name_token_key
         AND r.address_number = s.address_number
        WHERE s.address_number <> ''
        """
    )

    # ---------------------------------------------------------
    # R4 — first token + number + prefix
    #
    # This is deliberately stricter than the original V4 R4.
    # It reduces massive false candidate groups.
    # ---------------------------------------------------------

    run_rule(
        conn,
        "R4 first token + number + prefix",
        """
        INSERT OR IGNORE INTO candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM s1 s
        JOIN records r
          ON r.country = s.country
         AND r.first_name_token = s.first_name_token
         AND r.address_number = s.address_number
         AND r.name_prefix_sig = s.name_prefix_sig
        WHERE s.address_number <> ''
        """
    )

    # ---------------------------------------------------------
    # R5 — first token + number + suffix
    # ---------------------------------------------------------

    run_rule(
        conn,
        "R5 first token + number + suffix",
        """
        INSERT OR IGNORE INTO candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM s1 s
        JOIN records r
          ON r.country = s.country
         AND r.first_name_token = s.first_name_token
         AND r.address_number = s.address_number
         AND r.name_suffix_sig = s.name_suffix_sig
        WHERE s.address_number <> ''
        """
    )

    # ---------------------------------------------------------
    # R6 — exact address + first token
    # ---------------------------------------------------------

    run_rule(
        conn,
        "R6 exact address + first token",
        """
        INSERT OR IGNORE INTO candidates
        SELECT
            s.entity_id,
            r.entity_id
        FROM s1 s
        JOIN records r
          ON r.country = s.country
         AND r.address_key = s.address_key
         AND r.first_name_token = s.first_name_token
        WHERE s.address_key <> ''
        """
    )


def write_outputs(
    conn,
    matching_path,
    candidate_path
):
    print("Writing candidate_pairs.tsv...")

    start = time.time()

    with open(
        candidate_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(
            f,
            delimiter="\t",
            lineterminator="\n"
        )

        writer.writerow([
            "source1_entity_id",
            "candidate_entity_id"
        ])

        cursor = conn.execute("""
            SELECT
                source1_entity_id,
                candidate_entity_id
            FROM candidates
            ORDER BY source1_entity_id
        """)

        for row in cursor:
            writer.writerow(row)

    print(
        f"Candidate file written in "
        f"{(time.time() - start) / 60:.2f} min"
    )

    print("Writing matching_results.tsv...")

    start = time.time()

    with open(
        matching_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(
            f,
            delimiter="\t",
            lineterminator="\n"
        )

        writer.writerow([
            "source1_entity_id",
            "matched_entity_ids"
        ])

        cursor = conn.execute("""
            SELECT
                s.entity_id,
                COALESCE(
                    GROUP_CONCAT(
                        c.candidate_entity_id,
                        ','
                    ),
                    ''
                )
            FROM s1 s
            LEFT JOIN candidates c
              ON c.source1_entity_id = s.entity_id
            GROUP BY s.entity_id
            ORDER BY s.rowid
        """)

        for row in cursor:
            writer.writerow(row)

    print(
        f"Matching file written in "
        f"{(time.time() - start) / 60:.2f} min"
    )


def main(args):

    print("=" * 70)
    print("FAST V4 SUBMISSION GENERATOR")
    print("=" * 70)

    start_total = time.time()

    conn = sqlite3.connect(
        args.db
    )

    # Read/write performance
    conn.execute("PRAGMA cache_size=-500000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    create_s1_table(conn)

    processed = load_s1(
        conn,
        args.source1,
        args.limit
    )

    create_candidate_table(conn)

    generate_candidates(conn)

    candidate_count = conn.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    entities_with_candidates = conn.execute("""
        SELECT COUNT(DISTINCT source1_entity_id)
        FROM candidates
    """).fetchone()[0]

    print()
    print("=" * 70)
    print("CANDIDATE GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"S1 entities       : {processed:,}"
    )

    print(
        f"Total candidates   : {candidate_count:,}"
    )

    print(
        f"Entities with cand.: "
        f"{entities_with_candidates:,}"
    )

    print(
        f"Average candidates : "
        f"{candidate_count / processed:.2f}"
        if processed
        else
        "Average candidates : 0"
    )

    write_outputs(
        conn,
        args.matching,
        args.candidates
    )

    conn.close()

    elapsed = time.time() - start_total

    print()
    print("=" * 70)
    print("FAST V4 COMPLETE")
    print("=" * 70)

    print(
        f"Total time : {elapsed / 60:.2f} minutes"
    )

    print(
        f"Matching   : {args.matching}"
    )

    print(
        f"Candidates : {args.candidates}"
    )


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
        "--limit",
        type=int,
        default=0
    )

    args = parser.parse_args()

    main(args)
