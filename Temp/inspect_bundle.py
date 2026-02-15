from pathlib import Path
import UnityPy
bundle = Path(r'C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\AssetBundle\439583_CardArt_abf1baa1-4caff4c9340a5e7285d727d049086ce9.mtga')
env = UnityPy.load(bundle)
print('objects', len(env.objects))
for obj in env.objects:
    print(obj.type.name)
    if obj.type.name == 'Texture2D':
        tex = obj.read()
        img = tex.image
        print('texture name', tex.name, 'size', img.size)
        break
