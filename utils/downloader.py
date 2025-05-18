# utils/downloader.py
import requests
import os

def download_image(lat, lon, api_key, zoom=21, size="640x640", save_dir="downloads"):
    os.makedirs(save_dir, exist_ok=True)
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom={zoom}&size={size}&scale=2&maptype=satellite&key={api_key}"
    )
    fname = os.path.join(save_dir, f"roof_{lat}_{lon}.png")
    r = requests.get(url)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
