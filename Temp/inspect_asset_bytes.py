from pathlib import Path
path = Path(r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\AssetBundle\000168_CardArt_dad03855-3f3cd34ce1451f5cfb516b2768226ed3.mtga")
data = path.read_bytes()[:16]
print(data)
