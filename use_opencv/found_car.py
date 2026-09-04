# Search for a car in an image using OpenCV

import cv2

img = cv2.imread("img/vehicle.jpg")

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

edges = cv2.Canny(
    blur,
    50,
    150
)

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

for contour in contours:

    area = cv2.contourArea(contour)

    if area < 500:
        continue

    perimeter = cv2.arcLength(
        contour,
        True
    )

    approx = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    x, y, w, h = cv2.boundingRect(
        approx
    )

    vertices = len(approx)

    if vertices == 3:

        shape = "Triangle"

    elif vertices == 4:

        ratio = w / h

        if 0.95 <= ratio <= 1.05:
            shape = "Square"
        else:
            shape = "Rectangle"

    elif vertices > 6:

        shape = "Circle"

    else:

        shape = "Polygon"

    cv2.drawContours(
        img,
        [approx],
        -1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        img,
        shape,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

cv2.imshow("Detected Shapes", img)

print("Quantidade de contornos:", len(contours))

for contour in contours:

    area = cv2.contourArea(contour)

    if area < 50000:
        continue

    perimeter = cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    vertices = len(approx)

    print(
        f"Area={area:.0f} | "
        f"Vertices={vertices}"
    )

cv2.waitKey(0)
cv2.destroyAllWindows()