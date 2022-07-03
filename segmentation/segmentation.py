# import the necessary packages
from cProfile import label
from msilib.schema import Directory
from cv2 import cvtColor, threshold
import numpy as np
import argparse
import imutils
import cv2 as cv
from matplotlib import pyplot as plt

# Read the image and convert to black and white then equalize the image
img_original = cv.imread('segmentation/img/recorte3.jpg')
scale_percent_original = 40 # percent of original size
width_original = int(img_original.shape[1] * scale_percent_original / 100)
height_original = int(img_original.shape[0] * scale_percent_original / 100)
dim_original = (width_original, height_original)
img = cv.resize(img_original, dim_original, interpolation=cv.INTER_AREA)

# print(type(img))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hist_eq = cv.equalizeHist(gray)

cv.imshow('Gray', gray)
cv.imshow('Histogram', hist_eq)

# Threshold and binarization applying a threshold value at design time of value 18.5.
ret, threshold = cv.threshold(hist_eq, 15, 255, cv.THRESH_BINARY_INV)
cv.imshow('Binary', threshold)

# A kervel of elliptical shape with rows and colums multiples of 120
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (threshold.shape[0]//120,threshold.shape[1]//120))
# Dilatation with 1 iteration
dilation = cv.dilate(threshold,kernel,iterations = 1)
# Erosion with 2 iteration
erosion = cv.erode(dilation,kernel,iterations = 2)
# Dilation with 1 iteration
dilation2 = cv.dilate(erosion,kernel,iterations = 1)
# Show morphological image
cv.imshow("Morphological Transformation", dilation2)


# Region labeling
# Chose 4 for connectivity type
connectivity = 4
# apply connected component analysis to the thresholded image
output = cv.connectedComponentsWithStats(dilation2, connectivity, cv.CV_32S)
(numLabels, labels, stats, centroids) = output
# initialize an output mask to store all insects from the image
mask = np.zeros(dilation2.shape, dtype="uint8")
(numLabels, labels, stats, centroids) = output
# loop over the number of unique connected component labels
for i in range(0, numLabels):
	# if this is the first component then we examine the
	# *background* (typically we would just ignore this
	# component in our loop)
	if i == 0:
		text = "examining component {}/{} (background)".format(
			i + 1, numLabels)
	# otherwise, we are examining an actual connected component
	else:
		text = "examining component {}/{}".format( i + 1, numLabels)
	# print a status message update for the current connected
	# component
	print("[INFO] {}".format(text))
	# extract the connected component statistics and centroid for
	# the current label
	x = stats[i, cv.CC_STAT_LEFT]
	y = stats[i, cv.CC_STAT_TOP]
	w = stats[i, cv.CC_STAT_WIDTH]
	h = stats[i, cv.CC_STAT_HEIGHT]
	area = stats[i, cv.CC_STAT_AREA]
	(cX, cY) = centroids[i]
	# clone our original image (so we can draw on it) and then draw
	# a bounding box surrounding the connected component along with
	# a circle corresponding to the centroid
	output = dilation2.copy()
	cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
	cv.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
	# construct a mask for the current connected component by
	# finding a pixels in the labels array that have the current
	# connected component ID
	componentMask = (labels == i).astype("uint8") * 255
	# show our output image and connected component mask
	cv.imshow("Output", output)
	cv.imshow("Connected Component", componentMask)
	# Show each image section for every insect
	final_image = cv.bitwise_and(img, img, mask=componentMask)
	# Crop every label with a margin
	cropped_final = final_image[y:y+h+20,x:x+w+20]

	# Resize the image in 100% for better view
	scale_percent = 250 # percent of original size
	width = int(cropped_final.shape[1] * scale_percent / 100)
	height = int(cropped_final.shape[0] * scale_percent / 100)
	dim = (width, height)
	resize_cropped = cv.resize(cropped_final, dim, interpolation=cv.INTER_AREA)
	# Show each label corresponding to each insect detected
	cv.imshow('Final Image', resize_cropped)
	cv.waitKey(0)
	
cv.waitKey(0)


	
