import cv2 as cv
from RDP import  RDP
from Segmentation import Segmentation
from Test import Test


img = RDP("Root/Prueba/Grande2.png")
images = Segmentation(img)
print(len(images))
Resultados= Test(images)
print(Resultados)