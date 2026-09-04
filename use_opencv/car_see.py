# How IA can see the car in the image

import cv2

img = cv2.imread("color_detector/img/vehicle.jpg")

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
    100,
    200
)

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

for contour in contours:

    area = cv2.contourArea(contour)

    if area > 1000:

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        

cv2.drawContours(
    img,
    contours,
    -1,
    (0, 255, 0),
    2
)

cv2.imshow("Original", img)
cv2.imshow("Bordas", edges)

cv2.waitKey(0)

cv2.destroyAllWindows()