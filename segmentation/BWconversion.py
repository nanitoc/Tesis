# import the necessary packages
from msilib.schema import Directory
from cv2 import cvtColor
import numpy as np
import argparse
import imutils
import cv2 as cv
from matplotlib import pyplot as plt
import os

# Read the image and convert to black and white then equalize the image
img = cv.imread('segmentation/img/recorte.png')

# print(type(img))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hist_eq = cv.equalizeHist(gray)

cv.imshow('Gray', gray)
cv.imshow('Histogram', hist_eq)

 # Local threshold and binarization applying a threshold value at design time of value 7.
def custom_threshold(hist_eq):
    h, w = hist_eq.shape[:2]
    m = np.reshape(hist_eq, [1, w*h])
     # Mean
    mean = m.sum() / (w*h)
    # print('mean:', mean)
    ret, binary = cv.threshold(hist_eq, mean//(mean/7), 255, cv.THRESH_BINARY_INV)
    cv.imshow('binary', binary)
    cv.imwrite('segmentation/img/binary.png', binary)
    
custom_threshold(hist_eq)

# Morphological transformations
# Save the binary image to use it for morphological transformation
img_binary = cv.imread('segmentation/img/binary.png')

# A kervel of elliptical shape with rows and colums multiples of 120
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (img_binary.shape[0]//120,img_binary.shape[1]//120))
# Dilatation with 1 iteration
dilation = cv.dilate(img_binary,kernel,iterations = 1)
# Erosion with 2 iteration
erosion = cv.erode(dilation,kernel,iterations = 2)
# Dilation with 1 iteration
dilation2 = cv.dilate(erosion,kernel,iterations = 1)

cv.imshow("Morphological Transformation", dilation2)
cv.waitKey(0)