import cv2 as cv
from Root.RDP import  RDP
from Root.Segmentation import Segmentation
from Root.Test import Test


img = RDP("Root/Prueba/Trampa 10.png")
images = Segmentation(img)
Resultados = Test(images,32)
# print(Resultados)
n=len(images)   #numero de insectos en la trampa
Porcentaje=(Resultados.count('Alert')/n)*100 #porcentaje de diaphorina
                                 
print(f"La presencia de diaphorina es un {Porcentaje}% \nLa poblacion total en la trampa es: {n}")





