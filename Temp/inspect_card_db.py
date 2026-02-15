import sqlite3
from pathlib import Path
db=Path(r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw\Raw_CardDatabase_5f308e7a60516e3ac3d6c8ca9bbb638a.mtga")
con=sqlite3.connect(db)
cur=con.cursor()
print(cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
print(cur.execute("PRAGMA table_info('Cards')").fetchall())
print(cur.execute("PRAGMA table_info('Localizations_enUS')").fetchall())
row=cur.execute("SELECT GrpId, TitleId, ArtId FROM Cards LIMIT 1").fetchone();print('sample card', row)
con.close()
