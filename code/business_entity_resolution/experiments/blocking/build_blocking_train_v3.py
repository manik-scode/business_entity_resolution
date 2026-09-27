# src/blocking/build_blocking_train_v3.py

import sqlite3
import time
from pathlib import Path


SRC_DB = Path(r"results\blocking\blocking_train_v2.db")
OUT_DB = Path(r"results\blocking\blocking_train_v3.db")

R4_CAP = 500
R5_CAP = 500
R6_CAP = 250


def fmt(n):
    return f"{n:,}"


def main():
    start = time.time()

    print("=" * 80)
    print("BUILD BLOCKING TRAIN V3")
    print("=" * 80)

    if not SRC_DB.exists():
        raise FileNotFoundError(f"Source DB not found: {SRC_DB}")

    if OUT_DB.exists():
        raise FileExistsError(
            f"Output DB already exists:\n{OUT_DB}\n"
            "Delete/rename it manually if you intentionally want to rebuild."
        )

    OUT_DB.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Open source DB read-only
    # ---------------------------------------------------------

    src_uri = f"file:{SRC_DB.resolve().as_posix()}?mode=ro"

    src = sqlite3.connect(src_uri, uri=True)
    src.execute("PRAGMA query_only = ON")
    src.execute("PRAGMA temp_store = MEMORY")

    # ---------------------------------------------------------
    # Create output DB
    # ---------------------------------------------------------

    out = sqlite3.connect(str(OUT_DB))

    out.execute("PRAGMA journal_mode = WAL")
    out.execute("PRAGMA synchronous = NORMAL")
    out.execute("PRAGMA temp_store = MEMORY")

    print(f"\nSource DB : {SRC_DB}")
    print(f"Output DB : {OUT_DB}")

    # ---------------------------------------------------------
    # Inspect source
    # ---------------------------------------------------------

    s1_count = src.execute(
        "SELECT COUNT(*) FROM s1"
    ).fetchone()[0]

    source_candidate_count = src.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    print(f"\nSource S1 rows       : {fmt(s1_count)}")
    print(f"Existing candidates  : {fmt(source_candidate_count)}")

    # ---------------------------------------------------------
    # Create output schema
    # ---------------------------------------------------------

    out.execute("""
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

    out.execute("""
        CREATE TABLE candidates (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,
            PRIMARY KEY (
                source1_entity_id,
                candidate_entity_id
            )
        )
    """)

    out.execute("""
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # ---------------------------------------------------------
    # Copy S1
    # ---------------------------------------------------------

    print("\nCopying S1...")

    rows = src.execute("""
        SELECT
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
        FROM s1
    """)

    batch = []

    for row in rows:
        batch.append(row)

        if len(batch) >= 10000:
            out.executemany("""
                INSERT INTO s1 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            batch.clear()

    if batch:
        out.executemany("""
            INSERT INTO s1 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)

    out.commit()

    copied_s1 = out.execute(
        "SELECT COUNT(*) FROM s1"
    ).fetchone()[0]

    print(f"S1 copied: {fmt(copied_s1)}")

    # ---------------------------------------------------------
    # Copy existing V2 candidate set
    # ---------------------------------------------------------

    print("\nCopying existing V2 candidates...")

    src.execute("PRAGMA cache_size = -200000")

    cursor = src.execute("""
        SELECT
            source1_entity_id,
            candidate_entity_id
        FROM candidates
    """)

    batch = []
    copied = 0

    for row in cursor:
        batch.append(row)

        if len(batch) >= 50000:
            out.executemany("""
                INSERT OR IGNORE INTO candidates
                VALUES (?, ?)
            """, batch)

            copied += len(batch)
            batch.clear()

            if copied % 500000 == 0:
                print(f"  copied: {fmt(copied)}")

    if batch:
        out.executemany("""
            INSERT OR IGNORE INTO candidates
            VALUES (?, ?)
        """, batch)

    out.commit()

    after_v2 = out.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    print(f"V2 candidates in V3: {fmt(after_v2)}")

    # =========================================================
    # R4
    #
    # country + address_number + address_token_sig
    #
    # Keep only keys where the record-side frequency <= 500.
    # =========================================================

    print("\n" + "-" * 80)
    print(f"R4: country + address_number + address_token_sig")
    print(f"R4 frequency cap: {R4_CAP}")
    print("-" * 80)

    before = after_v2

    t = time.time()

    out.execute("""
        ATTACH DATABASE ? AS srcdb
    """, (str(SRC_DB.resolve()),))

    out.execute(f"""
        INSERT OR IGNORE INTO candidates (
            source1_entity_id,
            candidate_entity_id
        )
        SELECT
            s.entity_id,
            r.entity_id
        FROM s1 AS s
        JOIN srcdb.records AS r
          ON r.country = s.country
         AND r.address_number = s.address_number
         AND r.address_token_sig = s.address_token_sig
        JOIN (
            SELECT
                country,
                address_number,
                address_token_sig
            FROM srcdb.records
            WHERE address_number != ''
              AND address_token_sig != ''
            GROUP BY
                country,
                address_number,
                address_token_sig
            HAVING COUNT(*) <= {R4_CAP}
        ) AS k
          ON k.country = r.country
         AND k.address_number = r.address_number
         AND k.address_token_sig = r.address_token_sig
        WHERE s.address_number != ''
          AND s.address_token_sig != ''
    """)

    out.commit()

    after_r4 = out.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    print(f"R4 new candidates : {fmt(after_r4 - before)}")
    print(f"R4 total          : {fmt(after_r4)}")
    print(f"R4 runtime        : {time.time() - t:.1f}s")

    # =========================================================
    # R5
    #
    # country + first_name_token + address_token_sig
    #
    # Frequency cap = 500
    # =========================================================

    print("\n" + "-" * 80)
    print(f"R5: country + first_name_token + address_token_sig")
    print(f"R5 frequency cap: {R5_CAP}")
    print("-" * 80)

    before = after_r4

    t = time.time()

    out.execute(f"""
        INSERT OR IGNORE INTO candidates (
            source1_entity_id,
            candidate_entity_id
        )
        SELECT
            s.entity_id,
            r.entity_id
        FROM s1 AS s
        JOIN srcdb.records AS r
          ON r.country = s.country
         AND r.first_name_token = s.first_name_token
         AND r.address_token_sig = s.address_token_sig
        JOIN (
            SELECT
                country,
                first_name_token,
                address_token_sig
            FROM srcdb.records
            WHERE first_name_token != ''
              AND address_token_sig != ''
            GROUP BY
                country,
                first_name_token,
                address_token_sig
            HAVING COUNT(*) <= {R5_CAP}
        ) AS k
          ON k.country = r.country
         AND k.first_name_token = r.first_name_token
         AND k.address_token_sig = r.address_token_sig
        WHERE s.first_name_token != ''
          AND s.address_token_sig != ''
    """)

    out.commit()

    after_r5 = out.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    print(f"R5 new candidates : {fmt(after_r5 - before)}")
    print(f"R5 total          : {fmt(after_r5)}")
    print(f"R5 runtime        : {time.time() - t:.1f}s")

    # =========================================================
    # R6
    #
    # country + name_prefix_sig + address_number
    #
    # Frequency cap = 250
    # =========================================================

    print("\n" + "-" * 80)
    print(f"R6: country + name_prefix_sig + address_number")
    print(f"R6 frequency cap: {R6_CAP}")
    print("-" * 80)

    before = after_r5

    t = time.time()

    out.execute(f"""
        INSERT OR IGNORE INTO candidates (
            source1_entity_id,
            candidate_entity_id
        )
        SELECT
            s.entity_id,
            r.entity_id
        FROM s1 AS s
        JOIN srcdb.records AS r
          ON r.country = s.country
         AND r.name_prefix_sig = s.name_prefix_sig
         AND r.address_number = s.address_number
        JOIN (
            SELECT
                country,
                name_prefix_sig,
                address_number
            FROM srcdb.records
            WHERE name_prefix_sig != ''
              AND address_number != ''
            GROUP BY
                country,
                name_prefix_sig,
                address_number
            HAVING COUNT(*) <= {R6_CAP}
        ) AS k
          ON k.country = r.country
         AND k.name_prefix_sig = r.name_prefix_sig
         AND k.address_number = r.address_number
        WHERE s.name_prefix_sig != ''
          AND s.address_number != ''
    """)

    out.commit()

    after_r6 = out.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    print(f"R6 new candidates : {fmt(after_r6 - before)}")
    print(f"R6 total          : {fmt(after_r6)}")
    print(f"R6 runtime        : {time.time() - t:.1f}s")

    # ---------------------------------------------------------
    # Indexes
    # ---------------------------------------------------------

    print("\nCreating indexes...")

    out.execute("""
        CREATE INDEX idx_candidates_s1
        ON candidates(source1_entity_id)
    """)

    out.execute("""
        CREATE INDEX idx_candidates_candidate
        ON candidates(candidate_entity_id)
    """)

    out.execute("""
        CREATE INDEX idx_s1_country_prefix_number
        ON s1(country, name_prefix_sig, address_number)
    """)

    out.commit()

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    final_count = out.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    unique_s1 = out.execute(
        "SELECT COUNT(DISTINCT source1_entity_id) FROM candidates"
    ).fetchone()[0]

    out.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        [
            ("version", "v3"),
            ("base_db", str(SRC_DB)),
            ("base_candidates", str(source_candidate_count)),
            ("r4_cap", str(R4_CAP)),
            ("r5_cap", str(R5_CAP)),
            ("r6_cap", str(R6_CAP)),
            ("final_candidates", str(final_count)),
            ("candidate_s1_count", str(unique_s1)),
        ],
    )

    out.commit()

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("V3 BUILD COMPLETE")
    print("=" * 80)

    print(f"V2 base candidates : {fmt(after_v2)}")
    print(f"R4 added           : {fmt(after_r4 - after_v2)}")
    print(f"R5 added           : {fmt(after_r5 - after_r4)}")
    print(f"R6 added           : {fmt(after_r6 - after_r5)}")
    print(f"--------------------------------")
    print(f"FINAL candidates   : {fmt(final_count)}")
    print(f"Unique S1          : {fmt(unique_s1)}")
    print(f"Output DB          : {OUT_DB}")
    print(f"Total runtime      : {time.time() - start:.1f}s")

    print("\nSource DB was not modified.")
    print("Existing V2 artifacts remain untouched.")

    out.close()
    src.close()


if __name__ == "__main__":
    main()