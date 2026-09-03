# Automatically detects the dominant colors in an image and maps them to the closest CSS3 color names using KMeans clustering and webcolors library

import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors

img = cv2.imread('praia.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

pixels = hsv.reshape((-1, 3))
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

kmeans.fit(pixels)
colors = kmeans.cluster_centers_.astype(int)

labels = kmeans.labels_

counts = np.bincount(labels)

percentages = (counts / counts.sum()) * 100

result = list(zip(colors, percentages))

result.sort(key=lambda x: x[1], reverse=True)

def closest_color(rgb):
    min_distance = float("inf")
    closest_name = None

    for color_name, hex_value in webcolors._definitions._CSS3_NAMES_TO_HEX.items():

        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)

        distance = (
            (rgb[0] - r_c) ** 2 +
            (rgb[1] - g_c) ** 2 +
            (rgb[2] - b_c) ** 2
        )

        if distance < min_distance:
            min_distance = distance
            closest_name = color_name

    return closest_name


print("=== DOMINANT COLORS ===\n")

for i, (color, percentage) in enumerate(result, start=1):

    r, g, b = color

    color_name = closest_color((r, g, b))

    print(
        f"Color {i}: "
        f"{color_name.capitalize()} "
        f"(RGB({r}, {g}, {b})) "
        f"- {percentage:.2f}%"
    )