import cv2 as cv
from Root.RDP import  RDP
from Root.Segmentation import Segmentation
from Root.Test import Test


img = RDP("Root/Prueba/Grande4.png")
images = Segmentation(img)
print(len(images))
Resultados= Test(images)
print(Resultados)