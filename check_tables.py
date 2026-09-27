import sqlite3

for path in [
    r"results\blocking\blocking_train_v3.db",
    r"results\blocking\blocking_train_v2.db",
]:
    print("\nDB:", path)
    db = sqlite3.connect(path)
    rows = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    print(rows)
    db.close()
