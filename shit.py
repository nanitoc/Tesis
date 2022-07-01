from copyreg import pickle
import pickle
from traceback import print_tb
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import datetime


import time

Name= "Alert-No-alert-64x2-{}".format(int(time.time()))
#x="Alert-No-alert-64x2-{}".format(int(datetime.datetime()))
print(Name)