import UnityPy
bundle = r'C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\AssetBundle\104568_CardArt_21f02543-8f0f273679ff5dbe3a26d8e1717e8dc8.mtga'
env = UnityPy.load(bundle)
print('files', env.files.keys())
print('objects', len(env.objects))
for obj in env.objects:
    print(obj.type.name)
    if obj.type.name == 'Texture2D':
        tex = obj.read()
        print('texture', tex.name, tex.m_Width, tex.m_Height)
        tex.image.save('Temp/out.png')
        break
