import sqlite3
from pathlib import Path

DB_PATH = Path(r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw\Raw_CardDatabase_5f308e7a60516e3ac3d6c8ca9bbb638a.mtga")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# List all tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cur.fetchall()]
print("Tables:")
for t in tables:
    print("  ", t)

# For each table, show first few columns
for t in tables:
    print(f"\n=== {t} ===")
    cur.execute(f"PRAGMA table_info({t});")
    for cid, name, ctype, notnull, dflt, pk in cur.fetchall():
        print(f"  {name} ({ctype})")
