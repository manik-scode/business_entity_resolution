from pathlib import Path
from collections import Counter
import csv


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "student_resource" / "dataset"
TRAIN_ROOT = DATA_ROOT / "train"

FILES = {
    "source1": TRAIN_ROOT / "train_source1.tsv",
    "source2": TRAIN_ROOT / "train_source2.tsv",
    "source3": TRAIN_ROOT / "train_source3.tsv",
    "ground_truth": TRAIN_ROOT / "train_ground_truth.tsv",
}


# ---------------------------------------------------------
# INSPECT SOURCE FILE
# ---------------------------------------------------------

def inspect_source(name: str, path: Path):

    print("\n" + "=" * 70)
    print(f"{name.upper()}")
    print("=" * 70)

    print(f"File   : {path}")
    print(f"Exists : {path.exists()}")

    if not path.exists():
        return

    total_rows = 0
    missing_name = 0
    missing_address = 0
    missing_country = 0

    country_counter = Counter()

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
            errors="replace",
            newline=""
        ) as file:

            reader = csv.DictReader(file, delimiter="\t")

            print("\nColumns:")
            print(reader.fieldnames)

            if reader.fieldnames is None:
                print("ERROR: No header found.")
                return

            print("\nFirst 5 rows:")

            for row_number, row in enumerate(reader, start=1):

                total_rows += 1

                # Missing-value checks
                if not row.get("business_name", "").strip():
                    missing_name += 1

                if not row.get("business_address", "").strip():
                    missing_address += 1

                country = row.get("country", "").strip()

                if not country:
                    missing_country += 1
                    country_counter["(blank)"] += 1
                else:
                    country_counter[country] += 1

                # Print first 5 rows
                if row_number <= 5:
                    print(row)

    except Exception as exc:
        print(f"\nERROR while reading file:")
        print(exc)
        return

    print("\nTotal rows:", f"{total_rows:,}")

    print("\nMissing values:")
    print(f"{'business_name':17}: {missing_name:,}")
    print(f"{'business_address':17}: {missing_address:,}")
    print(f"{'country':17}: {missing_country:,}")

    print("\nCountries:")

    for country, count in country_counter.most_common():
        print(f"{country:15} {count:,}")


# ---------------------------------------------------------
# INSPECT GROUND TRUTH
# ---------------------------------------------------------

def inspect_ground_truth(path: Path):

    print("\n" + "=" * 70)
    print("GROUND TRUTH")
    print("=" * 70)

    print(f"File   : {path}")
    print(f"Exists : {path.exists()}")

    if not path.exists():
        return

    total_rows = 0
    match_distribution = Counter()
    no_match = 0
    maximum_matches = 0

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
            errors="replace",
            newline=""
        ) as file:

            reader = csv.DictReader(file, delimiter="\t")

            print("\nColumns:")
            print(reader.fieldnames)

            print("\nFirst 10 rows:")

            for row_number, row in enumerate(reader, start=1):

                total_rows += 1

                matched_ids = row.get(
                    "matched_entity_ids",
                    ""
                ).strip()

                if not matched_ids:
                    match_count = 0
                else:
                    match_count = len(
                        [
                            x for x in matched_ids.split(",")
                            if x.strip()
                        ]
                    )

                match_distribution[match_count] += 1

                if match_count == 0:
                    no_match += 1

                maximum_matches = max(
                    maximum_matches,
                    match_count
                )

                if row_number <= 10:
                    print(row)

    except Exception as exc:
        print(f"\nERROR while reading ground truth:")
        print(exc)
        return

    print("\nRows:", f"{total_rows:,}")

    print("\nMatch-count distribution:")

    for count, frequency in sorted(match_distribution.items()):
        print(
            f"{count:5} matches : "
            f"{frequency:,} entities"
        )

    print("\nSingleton / no-match S1 entities:")

    print(
        f"No-match entities    : "
        f"{no_match:,}"
    )

    if total_rows:
        print(
            f"No-match percentage  : "
            f"{no_match / total_rows * 100:.2f}%"
        )

    print("\nMaximum matches for one S1:")
    print(maximum_matches)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("AMAZON BUSINESS ENTITY RESOLUTION")
    print("DATASET INSPECTION")
    print("=" * 70)

    for name in ["source1", "source2", "source3"]:
        inspect_source(
            name,
            FILES[name]
        )

    inspect_ground_truth(
        FILES["ground_truth"]
    )

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)