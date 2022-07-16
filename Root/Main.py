import cv2 as cv
from RDP import  RDP
from Segmentation import Segmentation
from Test import Test


img = RDP("Root/Prueba/Forcemezcla.png")
images = Segmentation(img)
Resultados= Test(images)
print(Resultados)