import csv

PATH = r"student_resource\dataset\train\train_ground_truth.tsv"

with open(PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f, delimiter="\t")

    header = next(reader)
    first_row = next(reader)

print("HEADER:")
print(header)

print("\nFIRST ROW:")
print(first_row)