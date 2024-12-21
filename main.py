import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
#! part 1
img = cv.imread("images/emej.jpg")
cv.imshow("Image", img)
#! part 2
imge = cv.imread("images/emej.jpg",  cv.IMREAD_GRAYSCALE)
cv.imshow("Image grayscale", imge)
#! part 3
resized_input = cv.resize(imge, (1000, 1000))
hist = cv.calcHist([imge], [0], None, [256], [0, 256])
plt.figure()
plt.xlabel("Intensity")
plt.ylabel("Count Of Pixels")
plt.xlim([0, 256])
plt.locator_params(axis='x', nbins=50)
plt.plot(hist)
plt.show()
#! part 4..2  and part 6
Image = cv.imread("images/emej.jpg", cv.IMREAD_GRAYSCALE)

cv.imshow("Image Before gamma", Image)

gamma = 2.5
k = 255/np.float_power(np.max(Image), gamma)
rows, cols = Image.shape
for row in range(rows):
    for col in range(cols):
        Image[row][col] = k*np.float_power(Image[row][col], gamma)
output = np.array(Image, dtype='uint8')

cv.imshow("Image after gamma", output)
hist_new = cv.calcHist([output], [0], None, [256], [0, 256])
plt.figure()
plt.xlabel("Intensity")
plt.ylabel("Count Of Pixels")
plt.xlim([0, 256])
plt.locator_params(axis='x', nbins=50)
plt.plot(hist_new)
plt.show()


#! part 4..1 and part 6
Image1 = cv.imread("images/emej.jpg", cv.IMREAD_GRAYSCALE)
cv.imshow("image before lookup table", Image1)

lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
res = cv.LUT(resized_input, lookUpTable)

cv.imshow("image after lookup table", output)

hist_new = cv.calcHist([res], [0], None, [256], [0, 256])
plt.figure()
plt.xlabel("Intensity")
plt.ylabel("Count Of Pixels")
plt.xlim([0, 256])
plt.locator_params(axis='x', nbins=50)
plt.plot(hist_new)
plt.show()
cv.waitKey(0)

#! part 5
start = time.time()

#! Here write the code of lookup table or code of by pixel to calculate the run time

lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
res = cv.LUT(resized_input, lookUpTable)


end = time.time()
print(f"Runtime of the program is {end - start}")
