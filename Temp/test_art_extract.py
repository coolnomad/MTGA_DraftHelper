from scripts.pod_assets import CardAssetLoader
loader = CardAssetLoader()
print('card db', loader.card_db_path)
print('asset dir', loader.asset_dir)
rec = loader.find_by_name('Forest')
print('forest rec', rec)
if rec:
    uri = loader.art_uri_for_art_id(rec.art_id)
    print('art uri present', uri is not None)
else:
    print('no rec')
