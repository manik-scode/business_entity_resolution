import argparse
import csv
import re
import sqlite3
import unicodedata
from pathlib import Path


PUNCT_TABLE = str.maketrans(
    {chr(i): " " for i in range(0x10000) if unicodedata.category(chr(i)).startswith(("P", "S"))}
)

REPLACEMENTS = [
    ("private limited", "pvt ltd"),
    ("public limited", "pub ltd"),
    ("corporation", "corp"),
    ("company", "co"),
    ("incorporated", "inc"),
    ("limited", "ltd"),
    ("private", "pvt"),
    ("public", "pub"),
    ("societe anonyme", "sa"),
    ("societe", "soc"),
    ("etablissement", "etb"),
]

IGNORED_NAME_TERMS = {
    "corp", "co", "inc", "ltd", "pvt", "pub",
    "llc", "llp", "plc", "sa", "sarl", "gmbh",
    "ag", "limited", "company", "corporation",
}

ADDRESS_GENERIC = {
    "road", "rd", "street", "st", "lane", "ln",
    "avenue", "ave", "boulevard", "blvd",
    "highway", "hwy", "sector", "sec",
    "block", "district", "dist", "city",
    "town", "village", "near", "opposite",
    "opp", "india", "usa", "us", "france",
}


def normalize_text(value):
    value = unicodedata.normalize("NFKC", value or "")
    value = value.lower()
    value = value.translate(PUNCT_TABLE)

    for old, new in REPLACEMENTS:
        value = value.replace(old, new)

    value = re.sub(r"\s+", " ", value).strip()
    return value


def name_keys(name):
    normalized = normalize_text(name)

    tokens = [
        t for t in normalized.split()
        if t and t not in IGNORED_NAME_TERMS
    ]

    name_key = " ".join(tokens)
    sorted_name_key = " ".join(sorted(tokens))
    token_key = "|".join(sorted(set(tokens)))
    first_token = tokens[0] if tokens else ""

    compact = "".join(tokens)

    prefix = compact[:4] if compact else ""
    suffix = compact[-4:] if compact else ""
    length_bucket = (len(compact) // 5) * 5

    prefix_sig = f"{prefix}:{length_bucket}"
    suffix_sig = f"{suffix}:{length_bucket}"

    return {
        "name_key": name_key,
        "sorted_name_key": sorted_name_key,
        "name_token_key": token_key,
        "first_name_token": first_token,
        "name_prefix_sig": prefix_sig,
        "name_suffix_sig": suffix_sig,
    }


def address_keys(address):
    normalized = normalize_text(address)

    tokens = normalized.split()

    address_number_match = re.search(r"\b\d+[a-z]?\b", normalized)
    address_number = (
        address_number_match.group(0)
        if address_number_match
        else ""
    )

    informative = [
        t for t in tokens
        if t not in ADDRESS_GENERIC
        and len(t) >= 5
        and not re.fullmatch(r"\d+[a-z]?", t)
    ]

    if informative:
        informative_token = max(informative, key=len)
    else:
        informative_token = ""

    address_token_sig = informative_token[:5] if informative_token else ""

    return {
        "address_key": normalized,
        "address_number": address_number,
        "address_token_sig": address_token_sig,
    }


def get_rule_count(cur, country, column, value, extra_column=None, extra_value=None):
    if not value:
        return 0

    if extra_column:
        if not extra_value:
            return 0

        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM records
            WHERE country = ?
              AND {column} = ?
              AND {extra_column} = ?
            """,
            (country, value, extra_value),
        )
    else:
        cur.execute(
            f"""
            SELECT COUNT(*)
            FROM records
            WHERE country = ?
              AND {column} = ?
            """,
            (country, value),
        )

    return cur.fetchone()[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db", required=True)
    parser.add_argument("--s1", required=True)
    parser.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    s1_path = Path(args.s1).resolve()

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA cache_size=-131072")
    conn.execute("PRAGMA mmap_size=30000000000")
    cur = conn.cursor()

    print("=" * 100)
    print("ZERO-CANDIDATE DIAGNOSTIC")
    print("=" * 100)

    print(f"DB  : {db_path}")
    print(f"S1  : {s1_path}")
    print(f"Limit: {args.limit}")
    print()

    zero_count = 0
    checked = 0

    with s1_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            checked += 1

            entity_id = row["entity_id"]
            country_raw = (row["country"] or "").strip()

            COUNTRY_MAP = {
                "INDIA": "India",
                "US": "US",
                "FRANCE": "France",
            }

            country = COUNTRY_MAP.get(
                country_raw.upper(),
                country_raw
            )
            name = row["business_name"] or ""
            address = row["business_address"] or ""

            nk = name_keys(name)
            ak = address_keys(address)

            r1 = get_rule_count(
                cur,
                country,
                "name_key",
                nk["name_key"]
            )

            r2 = get_rule_count(
                cur,
                country,
                "sorted_name_key",
                nk["sorted_name_key"]
            )

            r3 = get_rule_count(
                cur,
                country,
                "name_token_key",
                nk["name_token_key"],
                "address_number",
                ak["address_number"]
            )

            r4 = get_rule_count(
                cur,
                country,
                "first_name_token",
                nk["first_name_token"],
                "address_number",
                ak["address_number"]
            )

            r5 = get_rule_count(
                cur,
                country,
                "address_key",
                ak["address_key"]
            )

            r6 = get_rule_count(
                cur,
                country,
                "name_prefix_sig",
                nk["name_prefix_sig"]
            )

            r7 = get_rule_count(
                cur,
                country,
                "name_suffix_sig",
                nk["name_suffix_sig"]
            )

            r8 = get_rule_count(
                cur,
                country,
                "address_token_sig",
                ak["address_token_sig"]
            )

            total = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8

            if total == 0:
                zero_count += 1

                print("=" * 100)
                print(f"ZERO CANDIDATE #{zero_count}")
                print("=" * 100)

                print(f"entity_id       : {entity_id}")
                print(f"country         : {country}")
                print(f"business_name   : {name}")
                print(f"business_address: {address}")
                print()

                print("--- NAME KEYS ---")
                for k, v in nk.items():
                    print(f"{k:20}: {v}")

                print()
                print("--- ADDRESS KEYS ---")
                for k, v in ak.items():
                    print(f"{k:20}: {v}")

                print()
                print("--- RULE COUNTS ---")
                print(f"R1 exact name                 : {r1}")
                print(f"R2 sorted name               : {r2}")
                print(f"R3 token + address number    : {r3}")
                print(f"R4 first token + number      : {r4}")
                print(f"R5 exact address             : {r5}")
                print(f"R6 prefix + length           : {r6}")
                print(f"R7 suffix + length           : {r7}")
                print(f"R8 address token             : {r8}")

                # Check whether this country exists at all.
                cur.execute(
                    "SELECT COUNT(*) FROM records WHERE country = ?",
                    (country,)
                )
                country_count = cur.fetchone()[0]

                print()
                print(f"Records for country '{country}': {country_count}")
                print()

                if zero_count >= args.limit:
                    break

    conn.close()

    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"S1 rows checked : {checked}")
    print(f"Zero candidates : {zero_count}")
    print("=" * 100)


if __name__ == "__main__":
    main()
