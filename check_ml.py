import sqlite3

db = sqlite3.connect(r"results\ml\ml_train_v1.db")

for table in ["features", "ml_pairs", "candidate_union"]:
    try:
        n = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {n:,}")
    except Exception as e:
        print(table, "ERROR:", e)

db.close()
