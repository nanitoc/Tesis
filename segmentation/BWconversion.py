# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2 as cv
from matplotlib import pyplot as plt

# Read the image and convert to black and white then equalize the image
img = cv.imread('segmentation/img/recorte.png')
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

custom_threshold(hist_eq)
cv.waitKey(0)

