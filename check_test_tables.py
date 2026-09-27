import sqlite3

db = sqlite3.connect(r"results\blocking\blocking_test_v2.db")
rows = db.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()
print(rows)
db.close()
