import sqlite3
from pathlib import Path
con=sqlite3.connect(r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw\Raw_CardDatabase_5f308e7a60516e3ac3d6c8ca9bbb638a.mtga")
cur=con.cursor()
rows=cur.execute("SELECT GrpId, ArtId, ArtPath FROM Cards LIMIT 5").fetchall()
for r in rows:
    print(r)
con.close()
