#Using the YOLOv8 model to detect objects in an image

from ultralytics import YOLO
import cv2

model = YOLO("yolov8m.pt")

results = model("img/transit.jpg")

for box in results[0].boxes:

    class_name = model.names[int(box.cls)]
    confidence = float(box.conf)

    print(class_name, confidence)

    cv2.imwrite(
        "results/transit_result.jpg",
        results[0].plot()
    )