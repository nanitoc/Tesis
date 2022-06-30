# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2 as cv
import matplotlib.pyplot as plt

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="img/trampa cromatica con fondo.jpg")
args = vars(ap.parse_args())

# load the image and display it
image = cv.imread(args["image"])
cv.imshow("Image", image)

# convert the image to grayscale and threshold it
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray, 110, 255,
	cv.THRESH_BINARY)[1]

# find the largest contour in the threshold image
cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv.contourArea)

# draw the shape of the contour on the output image, compute the
# bounding box, and display the number of points in the contour
output = image.copy()
img = cv.drawContours(output, [c], -1, (0, 255, 0), 3)
(x, y, w, h) = cv.boundingRect(c)

# Crop the image

cropped_image = img[y:y+h,x:x+w]
# show the original contour image

cv.imshow("Original Contour", cropped_image)
cv.waitKey(0)


