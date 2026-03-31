import os
import urllib.request

# สร้างโฟลเดอร์รอไว้
model_dir = "easyocr_models"
os.makedirs(model_dir, exist_ok=True)

urls = {
    "thai_g2.zip": "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/thai.zip",
    "english_g2.zip": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",
    "craft_mlt_25k.zip": "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip"
}

for name, url in urls.items():
    path = os.path.join(model_dir, name)
    if not os.path.exists(path):
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, path)
        # แนะนำให้หา Library มาแตก Zip ที่นี่ด้วย (เช่น zipfile)
