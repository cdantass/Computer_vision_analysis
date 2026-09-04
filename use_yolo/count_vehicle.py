# Can count the number of vehicles in an image using YOLOv8

from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")

results = model("img/airplane.jpg")

vehicle_classes = [
    "car",
    "motorcycle",
    "bus",
    "truck",
    "airplane",
]

vehicle_count = 0

for box in results[0].boxes:

    class_id = int(box.cls)
    confidence = float(box.conf)

    class_name = model.names[class_id]

    print(
        f"{class_name} - {confidence:.2%}"
    )

    if class_name in vehicle_classes:
        vehicle_count += 1

# Desenha as caixas do YOLO
annotated = results[0].plot()

# Escreve estatística
cv2.putText(
    annotated,
    f"Vehicles: {vehicle_count}",
    (10, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

# Salva imagem
cv2.imwrite(
    "results/airplane_result.jpg",
    annotated
)

print("\n=== RESULTADO ===")
print(f"Total de veiculos: {vehicle_count}")
