from ultralytics import YOLO
import os

model = YOLO("runs/detect/runs/face_mask_detection-2/weights/best.pt")

folder = "dataset/images/test"

for img in os.listdir(folder)[:10]:
    results = model(os.path.join(folder, img), conf=0.25)

    print("\n", img)

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(model.names[cls], conf)