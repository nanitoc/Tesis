import cv2 as cv
from cv2 import resize

img = cv.imread('Train/IMG1.jpg')
cv.imshow('D.Citri', img)

def rescaleFrame(frame, scale=0.75):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

resize_image = rescaleFrame(img)

cv.imshow('Image', resize_image)
cv.waitKey(0)

def changeRes(width,height):
    # Live Video
    capture.set(3,width)
    capture.set(4,height)

# Reading Videos
capture= cv.VideoCapture('Videos/VID1.mp4')

while True:
    isTrue, frame = capture.read()
    
    frame_resized = rescaleFrame(frame)

    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

