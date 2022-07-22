# import the necessary packages
import time
import pibooth
import libcamera
import cv2 as cv

from pibooth.utils import LOGGER
from pibooth.camera.base import BaseCamera
# initialize the camera and grab a reference to the raw camera capture
camera = BaseCamera()
rawCapture = PiRGBArray(camera)
# allow the camera to warmup
time.sleep(0.1)
# grab an image from the camera
camera.capture(rawCapture, format="rgb")
image = rawCapture.array
# display the image on screen and wait for a keypress
cv.imshow("Image", image)
cv.waitKey(0)