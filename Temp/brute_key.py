import UnityPy
from UnityPy.helpers import ArchiveStorageManager
from pathlib import Path
meta = Path(r'C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\il2cpp_data\Metadata\global-metadata.dat')
if not meta.exists():
    print('no metadata')
    quit()
key = ArchiveStorageManager.brute_force_key(meta, key_sig=b'b95e1eb85d5d90a5', data_sig=b'\x1b\x13\xbc\x00\x00\x00\x04CAB-59ac6')
print('key', key)
if key:
    UnityPy.set_assetbundle_decrypt_key(key)
    env = UnityPy.load(r'C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\AssetBundle\104568_CardArt_21f02543-8f0f273679ff5dbe3a26d8e1717e8dc8.mtga')
    print('objects', len(env.objects))
    for obj in env.objects:
        if obj.type.name == 'Texture2D':
            tex = obj.read(); print('texture', tex.name, tex.m_Width, tex.m_Height)
            tex.image.save('Temp/out.png')
            break
