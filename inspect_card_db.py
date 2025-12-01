import json
from collections import Counter
from pathlib import Path

import UnityPy

DB_PATH = Path(r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw\Raw_CardDatabase_5f308e7a60516e3ac3d6c8ca9bbb638a.mtga")

env = UnityPy.load(str(DB_PATH))

types = Counter(obj.type.name for obj in env.objects)
print("Object types and counts:")
for t, n in types.most_common():
    print(f"{t}: {n}")

# Look at a few TextAssets
print("\nSampling up to 5 TextAssets...")
count = 0
for obj in env.objects:
    if obj.type.name != "TextAsset":
        continue
    data = obj.read()
    if not hasattr(data, "text"):
        continue
    text = data.text
    print(f"\n--- TextAsset #{count} ---")
    print(text[:500])  # first 500 chars
    count += 1
    if count >= 5:
        break
