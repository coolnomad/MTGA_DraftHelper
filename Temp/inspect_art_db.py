import sqlite3
from pathlib import Path
art_path = Path(r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw\Raw_ArtCropDatabase_7b3c1ac0902ab51875925f432af0c9e7.mtga")
con = sqlite3.connect(art_path)
cur = con.cursor()
tables = [t[0] for t in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print('tables', tables)
for tbl in tables:
    cols = cur.execute(f"PRAGMA table_info('{tbl}')").fetchall()
    print(tbl, cols)
    sample = cur.execute(f"SELECT * FROM {tbl} LIMIT 1").fetchone()
    print('sample', sample)
con.close()
