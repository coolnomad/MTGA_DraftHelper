from pathlib import Path
from scripts.pod_assets import CardAssetLoader
loader = CardAssetLoader()
rec = loader.find_by_name('Forest')
uri = loader.art_uri_for_art_id(rec.art_id) if rec else None
print('uri len', len(uri) if uri else 0)
if uri:
    Path('Temp/forest_data_uri.txt').write_text(uri[:200], encoding='utf-8')
    print('wrote preview')
