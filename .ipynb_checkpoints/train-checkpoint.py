from ultralytics import YOLO

# Load pretrained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    workers=4,
    project="runs",
    name="face_mask_detection"
)