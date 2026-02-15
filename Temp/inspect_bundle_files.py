import UnityPy
from pathlib import Path
bundle = Path(r'C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\AssetBundle\104568_CardArt_21f02543-8f0f273679ff5dbe3a26d8e1717e8dc8.mtga')
env = UnityPy.load(bundle)
print('files', env.files.keys())
for f in env.files.values():
    print('file', f)
print('objects len', len(env.objects))
