# importar los paquetes necesarios
import numpy as np
import argparse
import imutils
import cv2 as cv

# construir el analizador de argumentos y analizar los argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="Prueba/Test1.jpeg")
args = vars(ap.parse_args())

# carga la imagen y la muestra
image = cv.imread(args["image"])
cv.imshow("Image", image)

# convertir la imagen a escala de grises y umbralizarla
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray, 110, 255,
	cv.THRESH_BINARY)[1]
cv.imshow("Thresh", thresh)

# encuentre el contorno más grande en la imagen del umbral
cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv.contourArea)

# dibuje la forma del contorno en la imagen de salida, calcule el cuadro
# delimitador y muestre el número de puntos en el contorno
output = image.copy()
img = cv.drawContours(output, [c], -1, (0, 255, 0), 3)
(x, y, w, h) = cv.boundingRect(c)

# recortar la imagen
cropped_image = img[y:y+h,x:x+w]

# mostrar la imagen de contorno original
cv.imshow("Original Contour", cropped_image)
cv.waitKey(0)


