from __future__ import annotations

import csv
import sqlite3
import time
from pathlib import Path

from rapidfuzz import fuzz


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_DB = (
    ROOT
    / "results"
    / "blocking"
    / "blocking_train_v2.db"
)

GROUND_TRUTH = (
    ROOT
    / "student_resource"
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

OUT_DIR = ROOT / "results" / "ml"

OUT_DB = OUT_DIR / "ml_train_v1.db"


# ============================================================
# CONFIG
# ============================================================

CAP = 500

NEG_MULTIPLIER = 5

MAX_NEGATIVES = 30

BATCH_SIZE = 10_000


# ============================================================
# HELPERS
# ============================================================

def now():
    return time.perf_counter()


def log(msg):
    print(f"[ML] {msg}", flush=True)


def connect_readonly(path: Path):
    uri = path.resolve().as_uri() + "?mode=ro"

    return sqlite3.connect(
        uri,
        uri=True,
        timeout=300,
    )


def attach_output(src_conn, output_path: Path):
    uri = output_path.resolve().as_uri()

    src_conn.execute(
        f"ATTACH DATABASE '{uri}' AS ml"
    )


def configure_output(conn):
    conn.execute(
        "PRAGMA synchronous=NORMAL"
    )

    conn.execute(
        "PRAGMA temp_store=FILE"
    )

    conn.execute(
        "PRAGMA cache_size=-200000"
    )


# ============================================================
# CREATE OUTPUT DB
# ============================================================

def create_output_db():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUT_DB.exists():

        log(
            f"Removing old ML DB: {OUT_DB}"
        )

        OUT_DB.unlink()

    conn = sqlite3.connect(
        OUT_DB,
        timeout=300
    )

    configure_output(conn)

    conn.executescript(
        """
        CREATE TABLE candidate_union (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,

            PRIMARY KEY (
                source1_entity_id,
                candidate_entity_id
            )
        );

        CREATE INDEX idx_candidate_union_s1
        ON candidate_union(source1_entity_id);

        CREATE INDEX idx_candidate_union_candidate
        ON candidate_union(candidate_entity_id);


        CREATE TABLE truth_pairs (
            source1_entity_id TEXT NOT NULL,
            matched_entity_id TEXT NOT NULL,

            PRIMARY KEY (
                source1_entity_id,
                matched_entity_id
            )
        );

        CREATE INDEX idx_truth_pairs_s1
        ON truth_pairs(source1_entity_id);


        CREATE TABLE ml_pairs (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,

            label INTEGER NOT NULL,
            pair_source TEXT NOT NULL,

            PRIMARY KEY (
                source1_entity_id,
                candidate_entity_id
            )
        );

        CREATE INDEX idx_ml_pairs_label
        ON ml_pairs(label);

        CREATE INDEX idx_ml_pairs_s1
        ON ml_pairs(source1_entity_id);


        CREATE TABLE features (
            source1_entity_id TEXT NOT NULL,
            candidate_entity_id TEXT NOT NULL,

            label INTEGER NOT NULL,
            pair_source TEXT NOT NULL,

            source TEXT,
            country TEXT,

            name_exact INTEGER,
            name_sorted_exact INTEGER,
            name_token_exact INTEGER,
            first_name_exact INTEGER,
            name_prefix_exact INTEGER,
            name_suffix_exact INTEGER,

            address_exact INTEGER,
            address_number_exact INTEGER,
            address_token_exact INTEGER,

            name_fuzz REAL,
            address_fuzz REAL,

            name_token_jaccard REAL,
            address_token_jaccard REAL,

            name_len_diff INTEGER,
            address_len_diff INTEGER,

            s1_name_missing INTEGER,
            candidate_name_missing INTEGER,

            s1_address_missing INTEGER,
            candidate_address_missing INTEGER,

            PRIMARY KEY (
                source1_entity_id,
                candidate_entity_id
            )
        );

        CREATE INDEX idx_features_label
        ON features(label);


        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )

    conn.commit()

    conn.close()

    log(
        f"Created fresh ML DB: {OUT_DB}"
    )


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_relevant_ground_truth(src_conn):

    log(
        "Reading 100k S1 IDs from blocking DB..."
    )

    s1_ids = {
        row[0]
        for row in src_conn.execute(
            """
            SELECT entity_id
            FROM main.s1
            """
        )
    }

    log(
        f"100k experiment S1 IDs loaded: "
        f"{len(s1_ids):,}"
    )


    insert_sql = """
        INSERT OR IGNORE INTO ml.truth_pairs
        (
            source1_entity_id,
            matched_entity_id
        )
        VALUES (?, ?)
    """


    total_rows = 0

    relevant_rows = 0

    batch = []


    with GROUND_TRUTH.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            s1_id = row[
                "source1_entity_id"
            ]

            if s1_id not in s1_ids:
                continue

            relevant_rows += 1

            raw = (
                row.get(
                    "matched_entity_ids"
                )
                or ""
            ).strip()


            if not raw:
                continue


            for candidate_id in raw.split(","):

                candidate_id = (
                    candidate_id.strip()
                )

                if not candidate_id:
                    continue


                batch.append(
                    (
                        s1_id,
                        candidate_id
                    )
                )

                total_rows += 1


                if len(batch) >= BATCH_SIZE:

                    src_conn.executemany(
                        insert_sql,
                        batch
                    )

                    src_conn.commit()

                    batch.clear()


        if batch:

            src_conn.executemany(
                insert_sql,
                batch
            )

            src_conn.commit()

            batch.clear()


    truth_count = src_conn.execute(
        """
        SELECT COUNT(*)
        FROM ml.truth_pairs
        """
    ).fetchone()[0]


    s1_with_truth = src_conn.execute(
        """
        SELECT COUNT(DISTINCT source1_entity_id)
        FROM ml.truth_pairs
        """
    ).fetchone()[0]


    log(
        f"Relevant GT S1 rows: "
        f"{relevant_rows:,}"
    )

    log(
        f"Truth pairs inserted: "
        f"{truth_count:,}"
    )

    log(
        f"S1 with >=1 true match: "
        f"{s1_with_truth:,}"
    )


# ============================================================
# BASE V2 CANDIDATES
# ============================================================

def build_base_candidates(src_conn):

    log(
        "Copying V2 base candidates..."
    )

    t0 = now()


    src_conn.execute(
        """
        INSERT OR IGNORE INTO ml.candidate_union
        (
            source1_entity_id,
            candidate_entity_id
        )

        SELECT
            source1_entity_id,
            candidate_entity_id

        FROM main.candidates
        """
    )

    src_conn.commit()


    count = src_conn.execute(
        """
        SELECT COUNT(*)
        FROM ml.candidate_union
        """
    ).fetchone()[0]


    log(
        f"V2 candidates: {count:,}"
        f" | {now() - t0:.1f}s"
    )


# ============================================================
# TARGETED R4 + R5
#
# R4:
# country + address_number + address_token_sig
#
# R5:
# country + first_name_token + address_token_sig
#
# Frequency cap = 500
# ============================================================

def add_r4_r5_targeted(src_conn):

    log(
        "Adding R4 + R5 using targeted 100k-key scan..."
    )

    t0 = now()


    # --------------------------------------------------------
    # STEP 1
    # Load keys belonging to 100k S1s
    # --------------------------------------------------------

    r4_s1_by_key = {}

    r5_s1_by_key = {}

    s1_count = 0


    cur = src_conn.execute(
        """
        SELECT
            entity_id,
            country,
            address_number,
            address_token_sig,
            first_name_token

        FROM main.s1
        """
    )


    for (
        s1_id,
        country,
        address_number,
        address_token_sig,
        first_name_token,
    ) in cur:

        s1_count += 1


        country = country or ""

        address_number = (
            address_number or ""
        )

        address_token_sig = (
            address_token_sig or ""
        )

        first_name_token = (
            first_name_token or ""
        )


        # -------------------------
        # R4
        # -------------------------

        if (
            country
            and address_number
            and address_token_sig
        ):

            key = (
                country,
                address_number,
                address_token_sig,
            )

            r4_s1_by_key.setdefault(
                key,
                []
            ).append(s1_id)


        # -------------------------
        # R5
        # -------------------------

        if (
            country
            and first_name_token
            and address_token_sig
        ):

            key = (
                country,
                first_name_token,
                address_token_sig,
            )

            r5_s1_by_key.setdefault(
                key,
                []
            ).append(s1_id)


    log(
        f"Target S1 rows: "
        f"{s1_count:,}"
    )

    log(
        f"Unique R4 keys: "
        f"{len(r4_s1_by_key):,}"
    )

    log(
        f"Unique R5 keys: "
        f"{len(r5_s1_by_key):,}"
    )


    # --------------------------------------------------------
    # STEP 2
    # Count frequencies
    # --------------------------------------------------------

    r4_counts = {}

    r5_counts = {}


    log(
        "Scanning records once for R4/R5 frequencies..."
    )


    scanned = 0


    cur = src_conn.execute(
        """
        SELECT
            country,
            address_number,
            address_token_sig,
            first_name_token

        FROM main.records
        """
    )


    for (
        country,
        address_number,
        address_token_sig,
        first_name_token,
    ) in cur:

        scanned += 1


        country = country or ""

        address_number = (
            address_number or ""
        )

        address_token_sig = (
            address_token_sig or ""
        )

        first_name_token = (
            first_name_token or ""
        )


        # -------------------------
        # R4
        # -------------------------

        if (
            country
            and address_number
            and address_token_sig
        ):

            key = (
                country,
                address_number,
                address_token_sig,
            )


            if key in r4_s1_by_key:

                current = r4_counts.get(
                    key,
                    0
                )


                if current <= CAP:

                    r4_counts[key] = (
                        current + 1
                    )


        # -------------------------
        # R5
        # -------------------------

        if (
            country
            and first_name_token
            and address_token_sig
        ):

            key = (
                country,
                first_name_token,
                address_token_sig,
            )


            if key in r5_s1_by_key:

                current = r5_counts.get(
                    key,
                    0
                )


                if current <= CAP:

                    r5_counts[key] = (
                        current + 1
                    )


        if scanned % 1_000_000 == 0:

            log(
                f"Frequency scan: "
                f"{scanned:,} records"
                f" | {now() - t0:.1f}s"
            )


    r4_allowed = {
        key
        for key, count
        in r4_counts.items()
        if count <= CAP
    }


    r5_allowed = {
        key
        for key, count
        in r5_counts.items()
        if count <= CAP
    }


    log(
        f"R4 allowed keys (<= {CAP}): "
        f"{len(r4_allowed):,}"
    )

    log(
        f"R5 allowed keys (<= {CAP}): "
        f"{len(r5_allowed):,}"
    )


    # --------------------------------------------------------
    # STEP 3
    # Generate candidates
    # --------------------------------------------------------

    log(
        "Scanning records second time "
        "to generate R4/R5 candidates..."
    )


    insert_sql = """
        INSERT OR IGNORE INTO ml.candidate_union
        (
            source1_entity_id,
            candidate_entity_id
        )
        VALUES (?, ?)
    """


    batch = []

    pair_count = 0

    scanned = 0


    cur = src_conn.execute(
        """
        SELECT
            entity_id,
            country,
            address_number,
            address_token_sig,
            first_name_token

        FROM main.records
        """
    )


    for (
        candidate_id,
        country,
        address_number,
        address_token_sig,
        first_name_token,
    ) in cur:

        scanned += 1


        country = country or ""

        address_number = (
            address_number or ""
        )

        address_token_sig = (
            address_token_sig or ""
        )

        first_name_token = (
            first_name_token or ""
        )


        # -------------------------
        # R4
        # -------------------------

        if (
            country
            and address_number
            and address_token_sig
        ):

            key = (
                country,
                address_number,
                address_token_sig,
            )


            if key in r4_allowed:

                for s1_id in r4_s1_by_key[key]:

                    batch.append(
                        (
                            s1_id,
                            candidate_id
                        )
                    )


        # -------------------------
        # R5
        # -------------------------

        if (
            country
            and first_name_token
            and address_token_sig
        ):

            key = (
                country,
                first_name_token,
                address_token_sig,
            )


            if key in r5_allowed:

                for s1_id in r5_s1_by_key[key]:

                    batch.append(
                        (
                            s1_id,
                            candidate_id
                        )
                    )


        if len(batch) >= 50_000:

            src_conn.executemany(
                insert_sql,
                batch
            )

            src_conn.commit()

            pair_count += len(batch)

            batch.clear()


        if scanned % 1_000_000 == 0:

            log(
                f"Candidate scan: "
                f"{scanned:,} records"
                f" | generated≈{pair_count:,}"
                f" | {now() - t0:.1f}s"
            )


    if batch:

        src_conn.executemany(
            insert_sql,
            batch
        )

        src_conn.commit()

        pair_count += len(batch)

        batch.clear()


    final_count = src_conn.execute(
        """
        SELECT COUNT(*)
        FROM ml.candidate_union
        """
    ).fetchone()[0]


    log(
        "R4 + R5 complete."
    )

    log(
        f"Candidate union: "
        f"{final_count:,}"
    )

    log(
        f"Total R4/R5 scan runtime: "
        f"{now() - t0:.1f}s"
    )


# ============================================================
# POSITIVE PAIRS
# ============================================================

def build_positive_pairs(src_conn):

    log(
        "Building positive ML pairs..."
    )


    src_conn.execute(
        """
        INSERT OR IGNORE INTO ml.ml_pairs
        (
            source1_entity_id,
            candidate_entity_id,
            label,
            pair_source
        )

        SELECT
            c.source1_entity_id,
            c.candidate_entity_id,
            1,
            'ground_truth'

        FROM ml.candidate_union AS c

        INNER JOIN ml.truth_pairs AS t
            ON t.source1_entity_id =
               c.source1_entity_id

           AND t.matched_entity_id =
               c.candidate_entity_id
        """
    )


    src_conn.commit()


    count = src_conn.execute(
        """
        SELECT COUNT(*)
        FROM ml.ml_pairs
        WHERE label = 1
        """
    ).fetchone()[0]


    log(
        f"Retrieved positive pairs: "
        f"{count:,}"
    )


# ============================================================
# HARD NEGATIVES
# ============================================================

def build_hard_negatives(src_conn):

    log(
        "Building hard negatives..."
    )

    t0 = now()


    src_conn.execute(
        """
        DROP TABLE IF EXISTS ml.pos_counts
        """
    )


    src_conn.execute(
        """
        CREATE TABLE ml.pos_counts AS

        SELECT
            source1_entity_id,
            COUNT(*) AS positive_count

        FROM ml.ml_pairs

        WHERE label = 1

        GROUP BY source1_entity_id
        """
    )


    src_conn.execute(
        """
        CREATE INDEX ml.idx_pos_counts_s1
        ON pos_counts(source1_entity_id)
        """
    )


    src_conn.execute(
        """
        WITH scored AS (

            SELECT
                c.source1_entity_id,
                c.candidate_entity_id,

                (
                    CASE
                        WHEN s.name_key <> ''
                         AND s.name_key = r.name_key
                        THEN 1 ELSE 0
                    END

                    +

                    CASE
                        WHEN s.sorted_name_key <> ''
                         AND s.sorted_name_key =
                             r.sorted_name_key
                        THEN 1 ELSE 0
                    END

                    +

                    CASE
                        WHEN s.name_token_key <> ''
                         AND s.name_token_key =
                             r.name_token_key
                        THEN 1 ELSE 0
                    END

                    +

                    CASE
                        WHEN s.first_name_token <> ''
                         AND s.first_name_token =
                             r.first_name_token
                        THEN 1 ELSE 0
                    END

                    +

                    CASE
                        WHEN s.name_prefix_sig <> ''
                         AND s.name_prefix_sig =
                             r.name_prefix_sig
                        THEN 1 ELSE 0
                    END

                    +

                    CASE
                        WHEN s.name_suffix_sig <> ''
                         AND s.name_suffix_sig =
                             r.name_suffix_sig
                        THEN 1 ELSE 0
                    END

                    +

                    CASE
                        WHEN s.address_key <> ''
                         AND s.address_key =
                             r.address_key
                        THEN 1 ELSE 0
                    END

                    +

                    CASE
                        WHEN s.address_number <> ''
                         AND s.address_number =
                             r.address_number
                        THEN 1 ELSE 0
                    END

                    +

                    CASE
                        WHEN s.address_token_sig <> ''
                         AND s.address_token_sig =
                             r.address_token_sig
                        THEN 1 ELSE 0
                    END

                ) AS block_score

            FROM ml.candidate_union AS c

            INNER JOIN main.s1 AS s
                ON s.entity_id =
                   c.source1_entity_id

            INNER JOIN main.records AS r
                ON r.entity_id =
                   c.candidate_entity_id

            LEFT JOIN ml.truth_pairs AS t
                ON t.source1_entity_id =
                   c.source1_entity_id

               AND t.matched_entity_id =
                   c.candidate_entity_id

            WHERE t.source1_entity_id IS NULL
        ),


        ranked AS (

            SELECT
                scored.*,

                ROW_NUMBER() OVER (
                    PARTITION BY
                        scored.source1_entity_id

                    ORDER BY
                        scored.block_score DESC,
                        scored.candidate_entity_id
                ) AS rn,


                CASE

                    WHEN COALESCE(
                        pc.positive_count,
                        0
                    ) > 0

                    THEN MIN(
                        COALESCE(
                            pc.positive_count,
                            0
                        ) * ?,
                        ?
                    )

                    ELSE ?

                END AS desired_negatives


            FROM scored


            LEFT JOIN ml.pos_counts AS pc

                ON pc.source1_entity_id =
                   scored.source1_entity_id


            WHERE scored.block_score > 0
        )


        INSERT OR IGNORE INTO ml.ml_pairs
        (
            source1_entity_id,
            candidate_entity_id,
            label,
            pair_source
        )

        SELECT
            source1_entity_id,
            candidate_entity_id,
            0,
            'hard_negative'

        FROM ranked

        WHERE rn <= desired_negatives
        """,

        (
            NEG_MULTIPLIER,
            MAX_NEGATIVES,
            MAX_NEGATIVES,
        )
    )


    src_conn.commit()


    total = src_conn.execute(
        """
        SELECT COUNT(*)
        FROM ml.ml_pairs
        """
    ).fetchone()[0]


    positives = src_conn.execute(
        """
        SELECT COUNT(*)
        FROM ml.ml_pairs
        WHERE label = 1
        """
    ).fetchone()[0]


    negatives = src_conn.execute(
        """
        SELECT COUNT(*)
        FROM ml.ml_pairs
        WHERE label = 0
        """
    ).fetchone()[0]


    log(
        f"ML pairs: {total:,}"
        f" | positives={positives:,}"
        f" | negatives={negatives:,}"
        f" | runtime={now() - t0:.1f}s"
    )


# ============================================================
# FUZZY FEATURES
# ============================================================

def safe_ratio(a, b):

    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b) / 100.0


def safe_token_jaccard(a, b):

    if not a or not b:
        return 0.0


    sa = set(a.split())

    sb = set(b.split())


    if not sa or not sb:
        return 0.0


    union = sa | sb


    if not union:
        return 0.0


    return len(sa & sb) / len(union)


# ============================================================
# GENERATE FEATURES
# ============================================================

def make_features(src_conn):

    log(
        "Generating ML features..."
    )

    t0 = now()


    insert_sql = """
        INSERT INTO ml.features
        (
            source1_entity_id,
            candidate_entity_id,

            label,
            pair_source,

            source,
            country,

            name_exact,
            name_sorted_exact,
            name_token_exact,
            first_name_exact,
            name_prefix_exact,
            name_suffix_exact,

            address_exact,
            address_number_exact,
            address_token_exact,

            name_fuzz,
            address_fuzz,

            name_token_jaccard,
            address_token_jaccard,

            name_len_diff,
            address_len_diff,

            s1_name_missing,
            candidate_name_missing,

            s1_address_missing,
            candidate_address_missing
        )

        VALUES
        (
            ?,?,?,?,?,?,

            ?,?,?,?,?,?,

            ?,?,?,

            ?,?,

            ?,?,

            ?,?,

            ?,?,?,?
        )
    """


    query = """
        SELECT

            p.source1_entity_id,
            p.candidate_entity_id,

            p.label,
            p.pair_source,

            r.source,
            s.country,

            s.name_key,
            r.name_key,

            s.sorted_name_key,
            r.sorted_name_key,

            s.name_token_key,
            r.name_token_key,

            s.first_name_token,
            r.first_name_token,

            s.name_prefix_sig,
            r.name_prefix_sig,

            s.name_suffix_sig,
            r.name_suffix_sig,

            s.address_key,
            r.address_key,

            s.address_number,
            r.address_number,

            s.address_token_sig,
            r.address_token_sig

        FROM ml.ml_pairs AS p

        INNER JOIN main.s1 AS s
            ON s.entity_id =
               p.source1_entity_id

        INNER JOIN main.records AS r
            ON r.entity_id =
               p.candidate_entity_id

        ORDER BY
            p.source1_entity_id,
            p.candidate_entity_id
    """


    cur = src_conn.execute(
        query
    )


    batch = []

    processed = 0


    for row in cur:

        (
            s1_id,
            candidate_id,

            label,
            pair_source,

            source,
            country,

            s_name,
            r_name,

            s_sorted,
            r_sorted,

            s_token,
            r_token,

            s_first,
            r_first,

            s_prefix,
            r_prefix,

            s_suffix,
            r_suffix,

            s_address,
            r_address,

            s_number,
            r_number,

            s_addr_token,
            r_addr_token,
        ) = row


        s_name = s_name or ""
        r_name = r_name or ""

        s_sorted = s_sorted or ""
        r_sorted = r_sorted or ""

        s_token = s_token or ""
        r_token = r_token or ""

        s_first = s_first or ""
        r_first = r_first or ""

        s_prefix = s_prefix or ""
        r_prefix = r_prefix or ""

        s_suffix = s_suffix or ""
        r_suffix = r_suffix or ""

        s_address = s_address or ""
        r_address = r_address or ""

        s_number = s_number or ""
        r_number = r_number or ""

        s_addr_token = (
            s_addr_token or ""
        )

        r_addr_token = (
            r_addr_token or ""
        )


        feature_row = (

            s1_id,
            candidate_id,

            int(label),
            pair_source,

            source,
            country,


            # -------------------------
            # NAME EXACT FEATURES
            # -------------------------

            int(
                bool(
                    s_name
                    and s_name == r_name
                )
            ),

            int(
                bool(
                    s_sorted
                    and s_sorted == r_sorted
                )
            ),

            int(
                bool(
                    s_token
                    and s_token == r_token
                )
            ),

            int(
                bool(
                    s_first
                    and s_first == r_first
                )
            ),

            int(
                bool(
                    s_prefix
                    and s_prefix == r_prefix
                )
            ),

            int(
                bool(
                    s_suffix
                    and s_suffix == r_suffix
                )
            ),


            # -------------------------
            # ADDRESS EXACT FEATURES
            # -------------------------

            int(
                bool(
                    s_address
                    and s_address == r_address
                )
            ),

            int(
                bool(
                    s_number
                    and s_number == r_number
                )
            ),

            int(
                bool(
                    s_addr_token
                    and s_addr_token == r_addr_token
                )
            ),


            # -------------------------
            # FUZZY FEATURES
            # -------------------------

            safe_ratio(
                s_name,
                r_name
            ),

            safe_ratio(
                s_address,
                r_address
            ),


            # -------------------------
            # TOKEN JACCARD
            # -------------------------

            safe_token_jaccard(
                s_token,
                r_token
            ),

            safe_token_jaccard(
                s_addr_token,
                r_addr_token
            ),


            # -------------------------
            # LENGTH FEATURES
            # -------------------------

            abs(
                len(s_name)
                -
                len(r_name)
            ),

            abs(
                len(s_address)
                -
                len(r_address)
            ),


            # -------------------------
            # MISSING FEATURES
            # -------------------------

            int(
                not bool(s_name)
            ),

            int(
                not bool(r_name)
            ),

            int(
                not bool(s_address)
            ),

            int(
                not bool(r_address)
            ),
        )


        batch.append(
            feature_row
        )

        processed += 1


        if len(batch) >= BATCH_SIZE:

            src_conn.executemany(
                insert_sql,
                batch
            )

            src_conn.commit()

            batch.clear()


            if processed % 100_000 == 0:

                log(
                    f"Features: "
                    f"{processed:,}"
                    f" | {now() - t0:.1f}s"
                )


    if batch:

        src_conn.executemany(
            insert_sql,
            batch
        )

        src_conn.commit()

        batch.clear()


    log(
        f"Feature generation complete: "
        f"{processed:,} rows"
        f" | {now() - t0:.1f}s"
    )


# ============================================================
# METADATA
# ============================================================

def write_metadata():

    conn = sqlite3.connect(
        OUT_DB,
        timeout=300
    )


    metadata = {

        "experiment":
            "100k_train_ml_v1",

        "source_blocking_db":
            str(TRAIN_DB),

        "ground_truth":
            str(GROUND_TRUTH),

        "base_blocking":
            "V2",

        "additional_rules":
            "R4,R5",

        "r4_rule":
            "country + address_number + address_token_sig",

        "r5_rule":
            "country + first_name_token + address_token_sig",

        "frequency_cap":
            str(CAP),

        "negative_multiplier":
            str(NEG_MULTIPLIER),

        "max_negatives_per_s1":
            str(MAX_NEGATIVES),

        "rapidfuzz":
            "enabled",
    }


    conn.executemany(
        """
        INSERT OR REPLACE INTO metadata
        (
            key,
            value
        )
        VALUES (?, ?)
        """,
        metadata.items(),
    )


    conn.commit()


    candidate_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM candidate_union
        """
    ).fetchone()[0]


    positive_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM ml_pairs
        WHERE label = 1
        """
    ).fetchone()[0]


    negative_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM ml_pairs
        WHERE label = 0
        """
    ).fetchone()[0]


    feature_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM features
        """
    ).fetchone()[0]


    conn.close()


    log("")

    log(
        "========================================"
    )

    log(
        "ML TRAIN DATASET COMPLETE"
    )

    log(
        "========================================"
    )

    log(
        f"Candidate pairs : "
        f"{candidate_count:,}"
    )

    log(
        f"Positive pairs  : "
        f"{positive_count:,}"
    )

    log(
        f"Negative pairs  : "
        f"{negative_count:,}"
    )

    log(
        f"Feature rows    : "
        f"{feature_count:,}"
    )

    log(
        f"Output DB       : "
        f"{OUT_DB}"
    )

    log(
        "========================================"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = now()


    if not TRAIN_DB.exists():

        raise FileNotFoundError(
            f"Blocking DB not found: {TRAIN_DB}"
        )


    if not GROUND_TRUTH.exists():

        raise FileNotFoundError(
            f"Ground truth not found: "
            f"{GROUND_TRUTH}"
        )


    create_output_db()


    src = connect_readonly(
        TRAIN_DB
    )


    try:

        configure_output(src)


        # ----------------------------------------------------
        # IMPORTANT
        #
        # main = blocking_train_v2.db
        # READ ONLY
        #
        # ml = separate writable ML DB
        # ----------------------------------------------------

        attach_output(
            src,
            OUT_DB
        )


        log(
            "Source DB attached READ-ONLY."
        )

        log(
            "ML DB attached as writable 'ml' database."
        )


        load_relevant_ground_truth(
            src
        )


        build_base_candidates(
            src
        )


        add_r4_r5_targeted(
            src
        )


        build_positive_pairs(
            src
        )


        build_hard_negatives(
            src
        )


        make_features(
            src
        )


    finally:

        try:

            src.execute(
                "DETACH DATABASE ml"
            )

        except Exception:

            pass


        src.close()


    write_metadata()


    log(
        f"TOTAL RUNTIME: "
        f"{(now() - total_start) / 60:.2f} minutes"
    )


if __name__ == "__main__":

    main()