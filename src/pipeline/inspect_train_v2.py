import sqlite3

DB_PATH = "results/blocking/blocking_train_v2.db"

conn = sqlite3.connect(DB_PATH)

print("=" * 70)
print("TRAIN V2 DATABASE INSPECTION")
print("=" * 70)

tables = conn.execute("""
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name
""").fetchall()

print("\nTables:")
for table in tables:
    print(" -", table[0])

for (table_name,) in tables:
    print("\n" + "-" * 70)
    print(f"TABLE: {table_name}")
    print("-" * 70)

    columns = conn.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    for col in columns:
        print(
            f"{col[1]:25} "
            f"type={col[2]:12} "
            f"pk={col[5]}"
        )

conn.close()