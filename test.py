from ultralytics import YOLO

# Load mô hình YOLOv8
model = YOLO("yolov8n.pt")  # Dùng mô hình nhẹ nhất

# Huấn luyện trên CPU
model.train(
    data="data.yaml",  # Đường dẫn dataset
    epochs=50,
    batch=8,           # Giảm batch size để tối ưu CPU
    imgsz=640,
    device="cpu"       # Quan trọng: Đặt rõ 'cpu' để tránh lỗi CUDA
)
