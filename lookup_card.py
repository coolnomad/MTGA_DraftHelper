import sqlite3
from pathlib import Path

DB_PATH = Path(r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw\Raw_CardDatabase_5f308e7a60516e3ac3d6c8ca9bbb638a.mtga")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

card_id = 97530  # change this to your grpId of interest

sql = """
SELECT c.GrpId, l.Loc
FROM Cards c
JOIN Localizations_enUS l
  ON c.TitleId = l.LocId
WHERE c.GrpId = ?
"""

cur.execute(sql, (card_id,))
row = cur.fetchone()
if row:
    print(f"GrpId {row[0]} → {row[1]}")
else:
    print("No card found for that GrpId")

conn.close()
