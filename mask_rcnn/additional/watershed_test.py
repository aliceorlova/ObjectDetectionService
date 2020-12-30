import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('coins.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
plt.subplot(121),plt.imshow(sure_bg),plt.title('sure_bg')
plt.xticks([]), plt.yticks([])
plt.show()
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
plt.subplot(121),plt.imshow(sure_fg),plt.title('sure_fg')
plt.xticks([]), plt.yticks([])
plt.show()
unknown = cv.subtract(sure_bg,sure_fg)
plt.subplot(121),plt.imshow(unknown),plt.title('unknown')
plt.xticks([]), plt.yticks([])
plt.show()
# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
plt.subplot(121),plt.imshow(markers),plt.title('markers')
plt.xticks([]), plt.yticks([])
plt.show()

markers = cv.watershed(img,markers)
plt.subplot(121),plt.imshow(markers),plt.title('markers2')
plt.xticks([]), plt.yticks([])
plt.show()

img[markers == -1] = [255,0,0]

plt.subplot(121),plt.imshow(img),plt.title('Final result')
plt.xticks([]), plt.yticks([])
plt.show()