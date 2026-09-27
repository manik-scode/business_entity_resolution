import sqlite3
import time
import csv
import re
import unicodedata
from pathlib import Path

DB = Path(r"results\blocking\blocking_train_full.db")
SOURCE1 = Path(r"student_resource\dataset\train\train_source1.tsv")


def normalize_text(value):
    if not value:
        return ""
    value = unicodedata.normalize("NFKC", value).lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def make_name_key(value):
    return normalize_text(value)


def make_sorted_name_key(value):
    text = normalize_text(value)
    return " ".join(sorted(text.split()))


def make_name_token_key(value):
    text = normalize_text(value)
    return " ".join(sorted(set(text.split())))


def first_token(value):
    text = normalize_text(value)
    return text.split()[0] if text else ""


def extract_address_number(value):
    text = normalize_text(value)
    match = re.search(r"\b\d+[a-z]?\b", text)
    return match.group(0) if match else ""


def create_s1_table(conn):
    conn.execute("DROP TABLE IF EXISTS s1")

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

    conn.commit()


def create_candidate_table(conn):
    conn.execute("DROP TABLE IF EXISTS candidates")

    conn.execute("""
    CREATE TABLE candidates (
        source1_entity_id TEXT,
        candidate_entity_id TEXT,
        PRIMARY KEY(source1_entity_id, candidate_entity_id)
    )
    """)

    conn.commit()


def load_s1(conn):
    insert_sql = """
    INSERT INTO s1 (
        entity_id,
        country,
        name_key,
        sorted_name_key,
        name_token_key,
        first_name_token,
        name_prefix_sig,
        name_suffix_sig,
        address_key,
        address_number,
        address_token_sig
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    batch = []
    total = 0

    with open(SOURCE1, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            entity_id = row["entity_id"]
            name = row.get("business_name") or ""
            address = row.get("business_address") or ""
            country = row.get("country") or ""

            name_norm = normalize_text(name)
            address_norm = normalize_text(address)

            tokens = name_norm.split()

            prefix = " ".join(tokens[:2])
            suffix = " ".join(tokens[-2:])

            address_tokens = sorted(set(address_norm.split()))
            address_token_sig = " ".join(address_tokens)

            batch.append((
                entity_id,
                country,
                name_norm,
                make_sorted_name_key(name),
                make_name_token_key(name),
                first_token(name),
                prefix,
                suffix,
                address_norm,
                extract_address_number(address),
                address_token_sig
            ))

            total += 1

            if len(batch) >= 10000:
                conn.executemany(insert_sql, batch)
                conn.commit()
                batch.clear()

                if total % 100000 == 0:
                    print("S1 loaded:", total)

    if batch:
        conn.executemany(insert_sql, batch)
        conn.commit()

    return total


def run_rule(conn, name, sql):
    before = conn.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    print("\nRunning", name)
    start = time.time()

    conn.execute(sql)
    conn.commit()

    after = conn.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    print(
        name,
        "new candidates:",
        after - before,
        "total:",
        after,
        "time:",
        round((time.time() - start) / 60, 2),
        "min"
    )


conn = sqlite3.connect(str(DB))

conn.execute("PRAGMA cache_size=-500000")
conn.execute("PRAGMA temp_store=MEMORY")
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")

start = time.time()

print("=" * 80)
print("FULL TRAIN BASE BLOCKING - R1/R2/R3")
print("=" * 80)

print("\nRebuilding S1...")
create_s1_table(conn)

count = load_s1(conn)

print("\nFULL S1 loaded:", count)

print("\nRebuilding candidates...")
create_candidate_table(conn)

run_rule(conn, "R1 exact name", """
INSERT OR IGNORE INTO candidates
SELECT s.entity_id, r.entity_id
FROM s1 s
JOIN records r
  ON r.country = s.country
 AND r.name_key = s.name_key
""")

run_rule(conn, "R2 sorted name", """
INSERT OR IGNORE INTO candidates
SELECT s.entity_id, r.entity_id
FROM s1 s
JOIN records r
  ON r.country = s.country
 AND r.sorted_name_key = s.sorted_name_key
""")

run_rule(conn, "R3 name token + number", """
INSERT OR IGNORE INTO candidates
SELECT s.entity_id, r.entity_id
FROM s1 s
JOIN records r
  ON r.country = s.country
 AND r.name_token_key = s.name_token_key
 AND r.address_number = s.address_number
WHERE s.address_number <> ''
""")

total = conn.execute(
    "SELECT COUNT(*) FROM candidates"
).fetchone()[0]

covered = conn.execute(
    "SELECT COUNT(DISTINCT source1_entity_id) FROM candidates"
).fetchone()[0]

print("\n" + "=" * 80)
print("FULL BASE COMPLETE")
print("=" * 80)

print("S1 entities        :", count)
print("Candidates         :", total)
print("S1 with candidates :", covered)
print("Average candidates :", round(total / count, 2))
print("Runtime minutes    :", round((time.time() - start) / 60, 2))

conn.close()
