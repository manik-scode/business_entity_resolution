import csv
import sqlite3
import time
from collections import defaultdict

DB_PATH = r"results\blocking\blocking_train_v2.db"
MISSED_PATH = r"results\blocking\ml_missed_pairs_v1.tsv"

CAPS = [25, 50, 100, 250, 500, 1000, 2000, 5000, None]


def normalize_key(value):
    if value is None:
        return ""
    return str(value)


def load_missed_pairs():
    pairs = []

    with open(MISSED_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            pairs.append(
                (
                    row["source1_entity_id"],
                    row["true_candidate_entity_id"],
                )
            )

    return pairs


def load_records(conn, entity_ids):
    records = {}

    ids = list(entity_ids)

    for start in range(0, len(ids), 5000):
        chunk = ids[start:start + 5000]

        placeholders = ",".join("?" * len(chunk))

        query = f"""
            SELECT
                entity_id,
                country,
                name_prefix_sig,
                name_suffix_sig,
                address_number
            FROM records
            WHERE entity_id IN ({placeholders})
        """

        for row in conn.execute(query, chunk):
            records[row[0]] = {
                "country": row[1],
                "name_prefix_sig": normalize_key(row[2]),
                "name_suffix_sig": normalize_key(row[3]),
                "address_number": normalize_key(row[4]),
            }

    return records


def build_frequency_maps(conn):
    print("Building frequency maps...")

    prefix_number = defaultdict(int)
    suffix_number = defaultdict(int)

    query = """
        SELECT
            country,
            name_prefix_sig,
            address_number,
            COUNT(*)
        FROM records
        WHERE name_prefix_sig != ''
          AND address_number != ''
        GROUP BY country, name_prefix_sig, address_number
    """

    for country, prefix, number, count in conn.execute(query):
        prefix_number[(country, prefix, number)] = count

    print(f"Prefix+number keys: {len(prefix_number):,}")

    query = """
        SELECT
            country,
            name_suffix_sig,
            address_number,
            COUNT(*)
        FROM records
        WHERE name_suffix_sig != ''
          AND address_number != ''
        GROUP BY country, name_suffix_sig, address_number
    """

    for country, suffix, number, count in conn.execute(query):
        suffix_number[(country, suffix, number)] = count

    print(f"Suffix+number keys: {len(suffix_number):,}")

    return prefix_number, suffix_number


def main():

    start_time = time.time()

    print("=" * 75)
    print("PREFIX+NUMBER / SUFFIX+NUMBER CAP TEST")
    print("=" * 75)

    print("\nLoading missed true pairs...")

    missed = load_missed_pairs()

    print(f"Missed true pairs: {len(missed):,}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA query_only = ON")

    # ---------------------------------------------------------
    # Load S1 records
    # ---------------------------------------------------------

    s1_ids = {x[0] for x in missed}

    print(f"Unique S1 IDs: {len(s1_ids):,}")

    print("\nLoading S1 records...")

    s1_records = {}

    ids = list(s1_ids)

    for start in range(0, len(ids), 5000):
        chunk = ids[start:start + 5000]

        placeholders = ",".join("?" * len(chunk))

        query = f"""
            SELECT
                entity_id,
                country,
                name_prefix_sig,
                name_suffix_sig,
                address_number
            FROM s1
            WHERE entity_id IN ({placeholders})
        """

        for row in conn.execute(query, chunk):
            s1_records[row[0]] = {
                "country": row[1],
                "name_prefix_sig": normalize_key(row[2]),
                "name_suffix_sig": normalize_key(row[3]),
                "address_number": normalize_key(row[4]),
            }

    print(f"S1 records loaded: {len(s1_records):,}")

    # ---------------------------------------------------------
    # Load true candidate records
    # ---------------------------------------------------------

    candidate_ids = {x[1] for x in missed}

    print("\nLoading candidate records...")

    candidate_records = load_records(conn, candidate_ids)

    print(f"Candidate records loaded: {len(candidate_records):,}")

    # ---------------------------------------------------------
    # Frequency maps
    # ---------------------------------------------------------

    prefix_freq, suffix_freq = build_frequency_maps(conn)

    print("\nTesting caps...")
    print("-" * 95)

    header = (
        f"{'CAP':>10}"
        f"{'PREFIX NEW':>15}"
        f"{'PREFIX REC':>15}"
        f"{'SUFFIX NEW':>15}"
        f"{'SUFFIX REC':>15}"
        f"{'COMBINED REC':>17}"
    )

    print(header)
    print("-" * 95)

    for cap in CAPS:

        prefix_new = 0
        suffix_new = 0

        prefix_recovered = set()
        suffix_recovered = set()

        # We only care about the currently missed TRUE pairs.
        for s1_id, candidate_id in missed:

            s1 = s1_records.get(s1_id)
            cand = candidate_records.get(candidate_id)

            if not s1 or not cand:
                continue

            country = s1["country"]

            # -------------------------------------------------
            # Prefix + address number
            # -------------------------------------------------

            prefix = s1["name_prefix_sig"]
            number = s1["address_number"]

            if prefix and number:
                key = (country, prefix, number)

                freq = prefix_freq.get(key, 0)

                if cap is None or freq <= cap:
                    prefix_recovered.add((s1_id, candidate_id))

            # -------------------------------------------------
            # Suffix + address number
            # -------------------------------------------------

            suffix = s1["name_suffix_sig"]

            if suffix and number:
                key = (country, suffix, number)

                freq = suffix_freq.get(key, 0)

                if cap is None or freq <= cap:
                    suffix_recovered.add((s1_id, candidate_id))

        # -----------------------------------------------------
        # Estimate new candidate count
        #
        # Count all S1 -> record combinations passing each key.
        # -----------------------------------------------------

        print(f"Calculating candidate volume for cap={cap}...")

        prefix_new = 0
        suffix_new = 0

        # Prefix rule
        query = """
            SELECT
                s1.country,
                s1.name_prefix_sig,
                s1.address_number,
                COUNT(*)
            FROM s1
            WHERE s1.name_prefix_sig != ''
              AND s1.address_number != ''
            GROUP BY
                s1.country,
                s1.name_prefix_sig,
                s1.address_number
        """

        for country, prefix, number, s1_count in conn.execute(query):

            freq = prefix_freq.get(
                (country, normalize_key(prefix), normalize_key(number)),
                0
            )

            if freq == 0:
                continue

            if cap is None or freq <= cap:
                prefix_new += s1_count * freq

        # Suffix rule
        query = """
            SELECT
                s1.country,
                s1.name_suffix_sig,
                s1.address_number,
                COUNT(*)
            FROM s1
            WHERE s1.name_suffix_sig != ''
              AND s1.address_number != ''
            GROUP BY
                s1.country,
                s1.name_suffix_sig,
                s1.address_number
        """

        for country, suffix, number, s1_count in conn.execute(query):

            freq = suffix_freq.get(
                (country, normalize_key(suffix), normalize_key(number)),
                0
            )

            if freq == 0:
                continue

            if cap is None or freq <= cap:
                suffix_new += s1_count * freq

        combined = prefix_recovered | suffix_recovered

        print(
            f"{str(cap) if cap is not None else 'UNCAPPED':>10}"
            f"{prefix_new:>15,}"
            f"{len(prefix_recovered):>15,}"
            f"{suffix_new:>15,}"
            f"{len(suffix_recovered):>15,}"
            f"{len(combined):>17,}"
        )

    conn.close()

    print("-" * 95)
    print(f"Runtime: {time.time() - start_time:.1f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()