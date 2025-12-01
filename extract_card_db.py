import json
from pathlib import Path

import UnityPy

# path to your Raw_CardDatabase file
DB_PATH = Path(r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw\Raw_CardDatabase_5f308e7a60516e3ac3d6c8ca9bbb638a.mtga")

env = UnityPy.load(str(DB_PATH))

card_entries = []

for obj in env.objects:
    if obj.type.name == "TextAsset":
        data = obj.read()
        if hasattr(data, "text") and "grpid" in data.text:
            print("FOUND CARD JSON TEXTASSET!")
            text = data.text
            try:
                parsed = json.loads(text)
                card_entries.append(parsed)
            except Exception:
                card_entries.append(text)

print("Found", len(card_entries), "possible card datasets")

# dump everything so we can inspect it
out_dir = Path(".")
for i, entry in enumerate(card_entries):
    out = out_dir / f"card_chunk_{i}.json"
    if isinstance(entry, str):
        out.write_text(entry, encoding="utf-8")
    else:
        with out.open("w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

print("Wrote card_chunk_*.json")
