# Using the YOLOv8 model to detect objects in an image

from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")

result = model("img/vehicle.jpg")

annotation = result[0].plot()

cv2.imwrite(
    "results/car_result.jpg",
    annotation
)

for box in result[0].boxes:

    class_id = int(box.cls)
    confidence = float(box.conf)

    class_name = model.names[class_id]

    print(
        f"{class_name} - {confidence:.2%}"
    )