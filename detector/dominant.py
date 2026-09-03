# Detecting colors in an image using OpenCV(manual dictionary of colors)

import cv2
import numpy as np

img = cv2.imread('praia.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

colors = {
    "red": [
        (np.array([0, 120, 70]), np.array([10, 255, 255])),
        (np.array([170, 120, 70]), np.array([180, 255, 255]))
    ],
    "green": [
        (np.array([36, 25, 25]), np.array([70, 255, 255]))
    ],
    "blue": [
        (np.array([94, 80, 2]), np.array([126, 255, 255]))
    ],
    "yellow": [
        (np.array([15, 150, 150]), np.array([35, 255, 255]))
    ],
    "orange": [
        (np.array([5, 50, 50]), np.array([15, 255, 255]))
    ],
    "purple": [
        (np.array([129, 50, 70]), np.array([158, 255, 255]))
    ],
    "pink": [
        (np.array([160, 50, 70]), np.array([170, 255, 255]))
    ],
    "white": [
        (np.array([0, 0, 200]), np.array([180, 20, 255]))
    ],
    "black": [
        (np.array([0, 0, 0]), np.array([180, 255, 30]))
    ],
    "gray": [
        (np.array([0, 0, 40]), np.array([180, 20, 200]))
    ],
    "brown": [
        (np.array([10, 100, 20]), np.array([20, 255, 200]))
    ]
}

height, width = img.shape[:2]
total_pixels = height * width

results = {}

for color_name, ranges in colors.items():
    mask = np.zeros((height, width), dtype=np.uint8)

    for lower, upper in ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(hsv, lower, upper)
        )

    color_pixels = cv2.countNonZero(mask)
    percentage = (color_pixels / total_pixels) * 100

    results[color_name] = percentage

print("=== COLORS FOUND ===\n")

for color, percentage in results.items():
    print(f"{color.capitalize()}: {percentage:.2f}%")

ordered_results = sorted(
    results.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\n=== DOMINANT COLORS ===\n")

for color, percentage in ordered_results:
    print(f"{color.capitalize()}: {percentage:.2f}%")

print(
    f"\nDominant color: "
    f"{ordered_results[0][0].capitalize()}"
)