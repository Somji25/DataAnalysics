import os
import urllib.request
import zipfile

# 1. สร้างโฟลเดอร์สำหรับเก็บโมเดล
model_dir = "easyocr_models"
os.makedirs(model_dir, exist_ok=True)

# 2. รายการ URL โมเดล (ปรับปรุง Link ให้ถูกต้องตามโครงสร้าง EasyOCR)
urls = {
    "thai_g2.zip": "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/thai.zip",
    "english_g2.zip": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",
    "craft_mlt_25k.zip": "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip"
}

for name, url in urls.items():
    zip_path = os.path.join(model_dir, name)
    # ชื่อไฟล์ .pth ที่ควรจะได้หลังจากแตก zip
    pth_name = name.replace(".zip", ".pth")
    pth_path = os.path.join(model_dir, pth_name)

    # ตรวจสอบว่ามีไฟล์ .pth หรือยัง (ถ้ามีแล้วไม่ต้องโหลดซ้ำ)
    if not os.path.exists(pth_path):
        print(f"📥 Downloading {name}...")
        try:
            # ดาวน์โหลดไฟล์ Zip
            urllib.request.urlretrieve(url, zip_path)
            
            # แตกไฟล์ Zip ทันที
            print(f"📦 Extracting {name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            
            # ลบไฟล์ Zip ทิ้งเพื่อประหยัดพื้นที่ Disk บน Server
            os.remove(zip_path)
            print(f"✅ {name} is ready.")
        except Exception as e:
            print(f"❌ Error processing {name}: {e}")
    else:
        print(f"ℹ️ {pth_name} already exists, skipping...")

print("🚀 All models are prepared!")
