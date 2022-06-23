from tkinter import dialog
import cv2 as cv
from cv2 import imshow
from cv2 import resize

img = cv.imread('Train/IMG1.jpg')

cv.imshow('D.Citri', img)

# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Blur
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)


#test
# Edge Cascade
canny = cv.Canny(blur, 90, 90)
cv.imshow('Canny Edges', canny)

# Dilating the image
dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow('Dilated', dilated)

# Eroding
eroded = cv.erode(dilated, (7,7), iterations=3)
imshow('Eroded', eroded)

# Resize
resized = cv.resize(img, (500,500),  interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)