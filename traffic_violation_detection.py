import cv2
import os
import csv
import numpy as np
import sqlite3
from sort import Sort  # Import thư viện SORT
from ultralytics import YOLO
import easyocr

# Khởi tạo mô hình YOLO và OCR
model = YOLO("yolov8n.pt")  # Thay bằng mô hình phù hợp
reader = easyocr.Reader(['en'])
tracker = Sort()  # Khởi tạo tracker SORT

# Thư mục lưu vi phạm
output_dir = "violations"
os.makedirs(output_dir, exist_ok=True)

# Kết nối hoặc tạo database SQLite
conn = sqlite3.connect("violations.db")
cursor = conn.cursor()

# Tạo bảng nếu chưa có
cursor.execute("""
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    obj_id INTEGER,
    license_plate TEXT,
    image_path TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# Tạo bộ nhớ lưu biển số đã nhận diện
plate_memory = {}

# Đọc video
video_path = '7.mp4'  
cap = cv2.VideoCapture(video_path)
width, height = int(cap.get(3)), int(cap.get(4))

# Vị trí vạch đỏ
red_line_y = int(height * 0.42)
line_thickness = 3

def save_violation(obj_id, plate_text, img_filename):
    cursor.execute("INSERT INTO violations (obj_id, license_plate, image_path) VALUES (?, ?, ?)", 
                   (obj_id, plate_text, img_filename))
    conn.commit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Dự đoán đối tượng
    results = model(frame)
    detections = []  # Danh sách phát hiện để đưa vào tracker

    # Vẽ vạch đỏ
    cv2.line(frame, (0, red_line_y), (width, red_line_y), (0, 0, 255), line_thickness)

    for result in results[0].boxes:
        xyxy = result.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        conf = result.conf[0].item()
        cls = result.cls[0].item()
        label = model.names[int(cls)]

        # Chỉ xử lý phương tiện
        allowed_classes = ["car", "motorcycle", "truck", "bus"]
        if label not in allowed_classes or conf < 0.5:
            continue

        # Thêm vào danh sách để SORT tracking
        detections.append([x1, y1, x2, y2, conf])

    # Chạy SORT tracking
    tracked_objects = tracker.update(np.array(detections))

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        vehicle_center_y = (y1 + y2) // 2
        
        if vehicle_center_y > red_line_y:  # Kiểm tra vi phạm
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"VIOLATION!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Kiểm tra nếu xe đã nhận diện biển số trước đó
            if obj_id in plate_memory:
                plate_text = plate_memory[obj_id]
            else:
                # Cắt vùng biển số để OCR
                vehicle_crop = frame[y1:y2, x1:x2]
                plate_results = reader.readtext(vehicle_crop)
                plate_text = None
                
                for (bbox, text, prob) in plate_results:
                    if prob > 0.5:
                        plate_text = text.replace(" ", "_")
                        plate_memory[obj_id] = plate_text  # Lưu biển số vào bộ nhớ
                        break
                
            if plate_text:
                cv2.putText(frame, plate_text, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                img_filename = f"{output_dir}/{plate_text}.jpg"
                cv2.imwrite(img_filename, frame)
                
                # Ghi vào database
                save_violation(obj_id, plate_text, img_filename)

    # Hiển thị video
    cv2.imshow("Traffic Violation Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Đóng kết nối database
conn.close()
