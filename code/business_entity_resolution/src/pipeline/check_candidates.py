import sqlite3

DB = "results/blocking/blocking_train_v2.db"

conn = sqlite3.connect(DB)

print("TABLES:")
for row in conn.execute("""
    SELECT name, sql
    FROM sqlite_master
    WHERE type = 'table'
"""):
    print(row)

print("\nINDEXES:")
for row in conn.execute("""
    SELECT name, tbl_name, sql
    FROM sqlite_master
    WHERE type = 'index'
"""):
    print(row)

conn.close()