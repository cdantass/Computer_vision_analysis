# Create a histogram of an image in RGB color space using OpenCV and Matplotlib

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('img/lua.jpg')   

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

colors = ('red', 'green', 'blue')

for i, col in enumerate(colors):
    hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

    plt.plot(hist, color=col)

plt.title('Histogram RGB')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.show()