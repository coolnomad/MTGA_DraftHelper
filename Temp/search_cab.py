from pathlib import Path
p=Path(r'C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\AssetBundle\104568_CardArt_21f02543-8f0f273679ff5dbe3a26d8e1717e8dc8.mtga')
data=p.read_bytes()
print(data[:20])
print(b'CAB-' in data)
print(data.find(b'CAB-'))
