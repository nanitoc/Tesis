# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('segmentation/img/recorte.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hist_eq = cv.equalizeHist(gray)

cv.imshow('Gray', gray)
cv.imshow('Histogram', hist_eq)
cv.waitKey(0)