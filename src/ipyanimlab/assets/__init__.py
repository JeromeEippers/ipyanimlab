import os.path

def get_asset_path(filename):
    
    folder = os.path.dirname(__file__)
    if not filename.lower().endswith('.usd'):
        filename += '.usd'
    return os.path.join(folder, filename)