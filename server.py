import os
import cv2
import uuid
import re
import datetime
import numpy as np
import easyocr
import collections
import base64
import gc      # สำหรับจัดการขยะใน RAM
import torch   # สำหรับคุม Memory ของ AI
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO
from thefuzz import process

app = Flask(__name__)

# ====== 1. ตั้งค่าฐานข้อมูล PostgreSQL & โมเดล ======

db_url = os.getenv('DATABASE_URL')
if db_url:
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://car_detection_db_user:PSf3eBnctkAHjigt6NejbtrCpuopVwL1@dpg-d75nnnruibrs73br3dkg-a.singapore-postgres.render.com/car_detection_db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# โหลดโมเดล AI และบังคับใช้ CPU (ประหยัด RAM กว่า GPU บน Server ฟรี)
MODEL_CAR_PATH = os.path.join("models", "best.pt")
MODEL_PLATE_PATH = os.path.join("models", "License.pt")

model_car = YOLO(MODEL_CAR_PATH) 
model_plate = YOLO(MODEL_PLATE_PATH)
model_car.to('cpu')
model_plate.to('cpu')

# โหลด EasyOCR (เน้นภาษาไทย/อังกฤษ) - โหลดครั้งเดียวตอน Start Server
reader = easyocr.Reader(['th', 'en'], gpu=False)

# ====== 2. นิยามโครงสร้าง Database (ORM) ======

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

# ====== 3. ตั้งค่าระบบไฟล์ & ข้อมูลจังหวัด ======
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static"
PROVINCES = [
    "กรุงเทพมหานคร", "สมุทรปราการ", "นนทบุรี", "ปทุมธานี", "พระนครศรีอยุธยา", "อ่างทอง", "ลพบุรี", "สิงห์บุรี", "ชัยนาท", "สระบุรี",
    "ชลบุรี", "ระยอง", "จันทบุรี", "ตราด", "ฉะเชิงเทรา", "ปราจีนบุรี", "นครนายก", "สระแก้ว",
    "นครราชสีมา", "บุรีรัมย์", "สุรินทร์", "ศรีสะเกษ", "อุบลราชธานี", "ยโสธร", "ชัยภูมิ", "อำนาจเจริญ", "บึงกาฬ", "หนองบัวลำภู", "ขอนแก่น", "อุดรธานี", "เลย", "หนองคาย", "มหาสารคาม", "ร้อยเอ็ด", "กาฬสินธุ์", "สกลนคร", "นครพนม", "มุกดาหาร",
    "เชียงใหม่", "ลำพูน", "ลำปาง", "อุตรดิตถ์", "แพร่", "น่าน", "พะเยา", "เชียงราย", "แม่ฮ่องสอน",
    "นครสวรรค์", "อุทัยธานี", "กำแพงเพชร", "ตาก", "สุโขทัย", "พิษณุโลก", "พิจิตร", "เพชรบูรณ์",
    "ราชบุรี", "กาญจนบุรี", "สุพรรณบุรี", "นครปฐม", "สมุทรสาคร", "สมุทรสงคราม", "เพชรบุรี", "ประจวบคีรีขันธ์",
    "นครศรีธรรมราช", "กระบี่", "พังงา", "ภูเก็ต", "สุราษฎร์ธานี", "ระนอง", "ชุมพร", "สงขลา", "สตูล", "ตรัง", "พัทลุง", "ปัตตานี", "ยะลา", "นราธิวาส"
]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ====== 4. ฟังก์ชันจัดการข้อมูล (Helper Functions) ======

def convert_cv2_to_base64(img_array):
    if img_array is None: return None
    try:
        _, buffer = cv2.imencode('.jpg', img_array)
        return base64.b64encode(buffer).decode('utf-8')
    except:
        return None

def advanced_thai_fixer(raw_text):
    if not raw_text: return "อ่านไม่ได้", ""
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
    
    province = ""
    if any(k in clean_raw for k in ["กรง", "เทพ", "นคร"]):
        province = "กรุงเทพมหานคร"
    else:
        match, score = process.extractOne(clean_raw, PROVINCES)
        province = match if score > 45 else "ไม่ทราบจังหวัด"
    return plate_no, province

def get_history():
    records = CarLog.query.order_by(CarLog.timestamp.desc()).limit(10).all()
    history = []
    for r in records:
        history.append({
            'Timestamp': r.timestamp.strftime("%d/%m/%Y %H:%M"),
            'CarType': r.car_type,
            'PlateNumber': r.plate_number,
            'Province': r.province,
            'ImageName': r.image_name,
            'FullImageBase64': r.full_image_base64,
            'PlateImageBase64': r.plate_image_base64
        })
    return history

def save_to_db(car_type, plate_no, province, img_name, full_img, plate_img):
    new_log = CarLog(
        car_type=car_type,
        plate_number=plate_no,
        province=province,
        image_name=img_name,
        full_image_base64=convert_cv2_to_base64(full_img),
        plate_image_base64=convert_cv2_to_base64(plate_img)
    )
    db.session.add(new_log)
    db.session.commit()

def update_db_record(img_name, new_car_type, new_plate, new_province):
    record = CarLog.query.filter_by(image_name=os.path.basename(img_name)).first()
    if record:
        record.car_type = new_car_type
        record.plate_number = new_plate
        record.province = new_province
        db.session.commit()
        return True
    return False

def run_ai_processing(img):
    # ลดขนาดภาพลงเล็กน้อยก่อนเข้า AI เพื่อประหยัด RAM (สำคัญมากสำหรับ Render)
    h, w = img.shape[:2]
    img_input = cv2.resize(img, (640, int(h * (640/w)))) if w > 640 else img

    with torch.no_grad(): # ปิด gradient เพื่อประหยัด RAM
        # ตรวจสอบรถ (ใช้ imgsz=320 เพื่อลดการใช้ RAM)
        res_car = model_car(img_input, imgsz=320, conf=0.7, verbose=False)
        
        class_mapping = {"Sedan": "รถเก๋ง", "SUV": "รถอเนกประสงค์", "Ambulance": "รถฉุกเฉิน", "Van": "รถตู้", "Pickup": "รถกระบะ"}
        car_info = []
        has_vehicle = False
        
        for r in res_car:
            if len(r.boxes) > 0:
                has_vehicle = True
                for box in r.boxes:
                    name = model_car.names[int(box.cls[0])]
                    car_info.append(class_mapping.get(name, name))
        
        if has_vehicle:
            plate_no, province, plate_crop_img = "", "", None
            # ตรวจสอบป้ายทะเบียน (imgsz=320)
            res_plate = model_plate(img_input, imgsz=320, conf=0.5, verbose=False)
            
            for r in res_plate:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    pad = 10
                    plate_crop = img_input[max(0, y1-pad):min(img_input.shape[0], y2+pad), 
                                           max(0, x1-pad):min(img_input.shape[1], x2+pad)]
                    if plate_crop.size > 0:
                        plate_crop_img = plate_crop
                        ocr_res = reader.readtext(plate_crop, detail=0, paragraph=True)
                        plate_no, province = advanced_thai_fixer("".join(ocr_res))
                        break
            
            # ล้างหน่วยความจำทันที
            del res_car, res_plate
            gc.collect()
            return car_info, plate_no, province, plate_crop_img
    
    gc.collect()
    return None

# ====== 5. เส้นทางหลัก (Routes) ======

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
    except:
        return jsonify({"vehicle_detected": False})

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
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, image_name), img)
        
        return jsonify({"status": "success", "filename": image_name})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/update-info", methods=["POST"])
def update_info():
    img_name = request.form.get("image_name")
    success = update_db_record(
        img_name, 
        request.form.get("car_type"), 
        request.form.get("plate_no"), 
        request.form.get("province")
    )
    return redirect(url_for('index', saved=1)) if success else ("Error", 500)

@app.route("/view/<filename>")
def view_history(filename):
    record = CarLog.query.filter_by(image_name=filename).first()
    if record:
        return render_template(
            "index.html", 
            image=record.image_name, 
            plate_zoom="zoom_"+record.image_name, 
            car_info=[record.car_type], 
            plate_no=record.plate_number, 
            province=record.province, 
            history=get_history()
        )
    return "Not Found", 404

@app.route("/dashboard")
def dashboard():
    sel_date = request.args.get('date') 
    query = CarLog.query
    if sel_date:
        query = query.filter(db.func.date(CarLog.timestamp) == sel_date)
    
    all_records = query.all()
    total = len(all_records)
    ct_counts = collections.Counter([r.car_type for r in all_records])
    prov_counts = collections.Counter([r.province for r in all_records])
    
    daily_query = db.session.query(
        db.func.date(CarLog.timestamp).label('date'), 
        db.func.count(CarLog.id).label('count')
    ).group_by(db.func.date(CarLog.timestamp)).order_by('date').all()
    
    chart = {
        "car_types": list(ct_counts.keys()), 
        "car_counts": list(ct_counts.values()), 
        "provinces": list(prov_counts.keys()), 
        "province_counts": list(prov_counts.values()), 
        "dates": [d.date.strftime("%d/%m/%Y") for d in daily_query], 
        "date_counts": [d.count for d in daily_query]
    }
    return render_template("dashboard.html", chart_data=chart, total=total, selected_date=sel_date)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
