from ultralytics import YOLO

model = YOLO("runs/detect/runs/face_mask_detection/weights/best.pt")

results = model("dataset/images/test/sample.jpg", show=True)

print(results)