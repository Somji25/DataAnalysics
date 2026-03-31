import os
import cv2
import uuid
import re
import datetime
import numpy as np
import easyocr
import collections
import base64
import gc
import torch
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO
from thefuzz import process

# บังคับ YOLO ทำงานแบบ Offline
os.environ['YOLO_OFFLINE'] = 'True'

app = Flask(__name__)

# ====== 1. ตั้งค่าฐานข้อมูล PostgreSQL ======
db_url = os.getenv('DATABASE_URL')
if db_url:
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://car_detection_db_user:PSf3eBnctkAHjigt6NejbtrCpuopVwL1@dpg-d75nnnruibrs73br3dkg-a.singapore-postgres.render.com/car_detection_db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ตัวแปรโมเดล (Lazy Loading)
model_car = None
model_plate = None
reader = None

def load_models():
    """โหลดโมเดลเมื่อจำเป็นเท่านั้น และบังคับใช้ CPU"""
    global model_car, model_plate, reader
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    if model_car is None:
        model_car = YOLO(os.path.join(BASE_DIR, "models", "best.pt")).to('cpu')
    if model_plate is None:
        model_plate = YOLO(os.path.join(BASE_DIR, "models", "License.pt")).to('cpu')
    if reader is None:
        # ระบุโฟลเดอร์เก็บโมเดลเพื่อป้องกันการดาวน์โหลดใหม่ทุกครั้ง
        model_storage_path = os.path.join(BASE_DIR, "easyocr_models")
        reader = easyocr.Reader(['th', 'en'], gpu=False, model_storage_directory=model_storage_path)

# ====== 2. โครงสร้าง Database (ORM) ======
class CarLog(db.Model):
    __tablename__ = 'car_logs'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.now)
    car_type = db.Column(db.String(50))
    plate_number = db.Column(db.String(50))
    province = db.Column(db.String(100))
    image_name = db.Column(db.String(100)) 
    full_image_base64 = db.Column(db.Text) 
    plate_image_base64 = db.Column(db.Text)

with app.app_context():
    db.create_all()

# ====== 3. ระบบไฟล์ & ข้อมูลจังหวัด ======
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static"
PROVINCES = [
    "กรุงเทพมหานคร", "สมุทรปราการ", "นนทบุรี", "ปทุมธานี", "พระนครศรีอยุธยา", "อ่างทอง", "ลพบุรี", "สิงห์บุรี", "ชัยนาท", "สระบุรี",
    "ชลบุรี", "ระยอง", "จันทบุรี", "ตราด", "ฉะเชิงเทรา", "ปราจีนบุรี", "นครนายก", "สระแก้ว",
    "นครราชสีมา", "บุรีรัยย์", "สุรินทร์", "ศรีสะเกษ", "อุบลราชธานี", "ยโสธร", "ชัยภูมิ", "อำนาจเจริญ", "บึงกาฬ", "หนองบัวลำภู", "ขอนแก่น", "อุดรธานี", "เลย", "หนองคาย", "มหาสารคาม", "ร้อยเอ็ด", "กาฬสินธุ์", "สกลนคร", "นครพนม", "มุกดาหาร",
    "เชียงใหม่", "ลำพูน", "ลำปาง", "อุตรดิตถ์", "แพร่", "น่าน", "พะเยา", "เชียงราย", "แม่ฮ่องสอน",
    "นครสวรรค์", "อุทัยธานี", "กำแพงเพชร", "ตาก", "สุโขทัย", "พิษณุโลก", "พิจิตร", "เพชรบูรณ์",
    "ราชบุรี", "กาญจนบุรี", "สุพรรณบุรี", "นครปฐม", "สมุทรสาคร", "สมุทรสงคราม", "เพชรบุรี", "ประจวบคีรีขันธ์",
    "นครศรีธรรมราช", "กระบี่", "พังงา", "ภูเก็ต", "สุราษฎร์ธานี", "ระนอง", "ชุมพร", "สงขลา", "สตูล", "ตรัง", "พัทลุง", "ปัตตานี", "ยะลา", "นราธิวาส"
]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ====== 4. Helper Functions ======
def convert_cv2_to_base64(img_array):
    if img_array is None: return None
    try:
        # ปรับ Quality เป็น 60 เพื่อประหยัดพื้นที่ Database
        _, buffer = cv2.imencode('.jpg', img_array, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        return base64.b64encode(buffer).decode('utf-8')
    except: return None

def advanced_thai_fixer(raw_text):
    if not raw_text: return "อ่านไม่ได้", "ไม่ทราบจังหวัด"
    clean_raw = re.sub(r'[^ก-ฮ0-9]', '', raw_text)
    numbers = re.findall(r'\d+', clean_raw)
    chars = re.findall(r'[ก-ฮ]+', clean_raw)
    chars_text = "".join(chars)
    plate_chars = chars_text[:2] if len(chars_text) >= 2 else chars_text

    if len(numbers) >= 2:
        plate_no = f"{numbers[0]} {plate_chars} {numbers[1]}"
    elif len(numbers) == 1:
        plate_no = f"{plate_chars} {numbers[0]}"
    else:
        plate_no = clean_raw if clean_raw else "อ่านไม่ได้"
    
    match, score = process.extractOne(clean_raw, PROVINCES)
    province = match if score > 45 else "ไม่ทราบจังหวัด"
    if any(k in clean_raw for k in ["กรง", "เทพ", "นคร"]): province = "กรุงเทพมหานคร"
    
    return plate_no, province

def save_to_db(car_type, plate_no, province, img_name, full_img, plate_img):
    new_log = CarLog(
        car_type=car_type, plate_number=plate_no, province=province,
        image_name=img_name,
        full_image_base64=convert_cv2_to_base64(full_img),
        plate_image_base64=convert_cv2_to_base64(plate_img)
    )
    db.session.add(new_log)
    db.session.commit()

def get_history():
    records = CarLog.query.order_by(CarLog.timestamp.desc()).limit(10).all()
    return [{
        'Timestamp': r.timestamp.strftime("%d/%m/%Y %H:%M"),
        'CarType': r.car_type, 'PlateNumber': r.plate_number,
        'Province': r.province, 'ImageName': r.image_name,
        'FullImageBase64': r.full_image_base64,
        'PlateImageBase64': r.plate_image_base64
    } for r in records]

# ====== 5. AI Processing (Optimized for Render) ======
def run_ai_processing(img):
    load_models()
    h, w = img.shape[:2]
    # Resize ภาพเข้า AI ให้เล็กลง (320-480px) เพื่อป้องกัน RAM เต็ม
    img_input = cv2.resize(img, (480, int(h * (480/w)))) if w > 480 else img

    car_info = []
    has_vehicle = False
    
    with torch.no_grad():
        res_car = model_car(img_input, imgsz=320, conf=0.7, verbose=False)
        if len(res_car) > 0 and len(res_car[0].boxes) > 0:
            has_vehicle = True
            class_mapping = {"Sedan": "รถเก๋ง", "SUV": "รถอเนกประสงค์", "Ambulance": "รถฉุกเฉิน", "Van": "รถตู้", "Pickup": "รถกระบะ"}
            for box in res_car[0].boxes:
                name = model_car.names[int(box.cls[0])]
                car_info.append(class_mapping.get(name, name))
        
        # ล้าง RAM หลังรันโมเดลแรก
        del res_car
        gc.collect()

        if has_vehicle:
            plate_no, province, plate_crop_img = "", "", None
            res_plate = model_plate(img_input, imgsz=320, conf=0.5, verbose=False)
            
            for r in res_plate:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pad = 10
                    plate_crop = img_input[max(0, y1-pad):min(img_input.shape[0], y2+pad), 
                                           max(0, x1-pad):min(img_input.shape[1], x2+pad)]
                    if plate_crop.size > 0:
                        plate_crop_img = plate_crop
                        # ลดขนาดภาพป้ายทะเบียนก่อนส่งให้ OCR เพื่อความเร็ว
                        ocr_res = reader.readtext(cv2.resize(plate_crop, (0,0), fx=0.8, fy=0.8), detail=0)
                        plate_no, province = advanced_thai_fixer("".join(ocr_res))
                        break
            
            del res_plate
            gc.collect()
            return car_info, plate_no, province, plate_crop_img
            
    return None

# ====== 6. Routes ======
@app.route("/", methods=["GET", "POST"])
def index():
    image_name, plate_zoom_name, car_info = None, None, []
    plate_no, province = "", ""
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            temp_name = str(uuid.uuid4()) + ".jpg"
            upload_path = os.path.join(UPLOAD_FOLDER, temp_name)
            file.save(upload_path)
            img = cv2.imread(upload_path)
            if img is not None:
                result = run_ai_processing(img)
                if result:
                    car_info, plate_no, province, plate_crop_img = result
                    image_name = temp_name
                    if plate_crop_img is not None:
                        plate_zoom_name = "zoom_" + image_name
                        cv2.imwrite(os.path.join(OUTPUT_FOLDER, plate_zoom_name), plate_crop_img)
                    
                    final_car = car_info[0] if car_info else "ไม่ระบุ"
                    save_to_db(final_car, plate_no, province, image_name, img, plate_crop_img)
                    cv2.imwrite(os.path.join(OUTPUT_FOLDER, image_name), img)
                else:
                    if os.path.exists(upload_path): os.remove(upload_path)

    return render_template("index.html", image=image_name, plate_zoom=plate_zoom_name, car_info=car_info, plate_no=plate_no, province=province, history=get_history())

@app.route("/detect-vehicle-only", methods=["POST"])
def detect_vehicle_only():
    try:
        load_models()
        data = request.json
        image_data = data.get("image")
        if not image_data or "," not in image_data: return jsonify({"vehicle_detected": False})
        encoded = image_data.split(",", 1)[1]
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        with torch.no_grad():
            results = model_car(img, imgsz=320, conf=0.75, verbose=False)
            detected = any(len(r.boxes) > 0 for r in results)
        
        gc.collect()
        return jsonify({"vehicle_detected": detected})
    except: return jsonify({"vehicle_detected": False})

@app.route("/upload-base64", methods=["POST"])
def upload_base64():
    try:
        data = request.json
        image_data = data.get("image")
        if not image_data or "," not in image_data: return jsonify({"status": "no_vehicle_detected"})
        encoded = image_data.split(",", 1)[1]
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        result = run_ai_processing(img)
        if result is None: return jsonify({"status": "no_vehicle_detected"})

        car_info, plate_no, province, plate_crop_img = result
        image_name = str(uuid.uuid4()) + ".jpg"
        
        if plate_crop_img is not None:
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, "zoom_" + image_name), plate_crop_img)
        
        final_car = car_info[0] if car_info else "ไม่ระบุ"
        save_to_db(final_car, plate_no, province, image_name, img, plate_crop_img)
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, image_name), img)
        
        return jsonify({"status": "success", "filename": image_name})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})

@app.route("/update-info", methods=["POST"])
def update_info():
    img_name = request.form.get("image_name")
    record = CarLog.query.filter_by(image_name=os.path.basename(img_name)).first()
    if record:
        record.car_type = request.form.get("car_type")
        record.plate_number = request.form.get("plate_no")
        record.province = request.form.get("province")
        db.session.commit()
        return redirect(url_for('index', saved=1))
    return ("Error", 500)

@app.route("/view/<filename>")
def view_history(filename):
    record = CarLog.query.filter_by(image_name=filename).first()
    if record:
        return render_template("index.html", image=record.image_name, plate_zoom="zoom_"+record.image_name, car_info=[record.car_type], plate_no=record.plate_number, province=record.province, history=get_history())
    return "Not Found", 404

@app.route("/dashboard")
def dashboard():
    sel_date = request.args.get('date') 
    query = CarLog.query
    if sel_date: query = query.filter(db.func.date(CarLog.timestamp) == sel_date)
    
    all_records = query.all()
    ct_counts = collections.Counter([r.car_type for r in all_records])
    prov_counts = collections.Counter([r.province for r in all_records])
    
    daily_query = db.session.query(db.func.date(CarLog.timestamp).label('date'), db.func.count(CarLog.id).label('count')).group_by(db.func.date(CarLog.timestamp)).order_by('date').all()
    
    chart = {
        "car_types": list(ct_counts.keys()), "car_counts": list(ct_counts.values()),
        "provinces": list(prov_counts.keys()), "province_counts": list(prov_counts.values()),
        "dates": [d.date.strftime("%d/%m/%Y") for d in daily_query], "date_counts": [d.count for d in daily_query]
    }
    return render_template("dashboard.html", chart_data=chart, total=len(all_records), selected_date=sel_date)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
