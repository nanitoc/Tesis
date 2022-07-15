# importar los paquetes necesarios
from cProfile import label
from msilib.schema import Directory
from cv2 import cvtColor, threshold
import numpy as np
import argparse
import imutils
import cv2 as cv
import glob
from matplotlib import pyplot as plt

# leer la imagen y convertirla a blanco y negro, luego ecualizar la imagen
img_original = cv.imread("Root\Prueba\Force_crop1.png")
scale_percent_original = 100 # percent of original size
width_original = int(img_original.shape[1] * scale_percent_original / 100)
height_original = int(img_original.shape[0] * scale_percent_original / 100)
dim_original = (width_original, height_original)
img = cv.resize(img_original, dim_original, interpolation=cv.INTER_AREA)

# convertir a escalas de grises
gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)

# calcular histogramas
hist = cv.calcHist([img_original], [0], None, [256], [0, 256])
# plt.plot(hist, color='gray' )
# plt.xlabel('intensidad de iluminacion')
# plt.ylabel('cantidad de pixeles')
# plt.show()

# ecualizar histograma
hist_eq = cv.equalizeHist(gray)

# mostrar histogramas
# hist = cv.calcHist([hist_eq], [0], None, [256], [0, 256])
# plt.plot(hist, color='gray' )
# plt.xlabel('intensidad de iluminacion')
# plt.ylabel('cantidad de pixeles')
# plt.show()

# mostrar imágenes en escala de grises y ecualizadas
cv.imshow("Gray", gray)
cv.imshow("Histogram", hist_eq)

# umbral y binarización aplicando un valor de umbral en tiempo de diseño de valor 15.
ret, threshold = cv.threshold(hist_eq, 15, 255, cv.THRESH_BINARY_INV)
cv.imshow("Binary", threshold)

# un núcleo de forma elíptica con filas y columnas múltiplos de 120
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (threshold.shape[0]//120,threshold.shape[1]//120))
# dilatación con 1 iteración
dilatation = cv.dilate(threshold,kernel,iterations = 5)
cv.imshow("Dilatation", dilatation)
# erosión con 2 iteraciones
erosion = cv.erode(dilatation,kernel,iterations = 2)
cv.imshow("Erosion", erosion)
# dilatación con 1 iteración 
dilatation2 = cv.dilate(erosion,kernel,iterations = 1)
# Mostrar imagen de transformaciones morfológicas
cv.imshow("Morphological Transformation", dilatation2)


# etiquetado de regiones
# elija 4 para el tipo de conectividad 
connectivity = 4
# aplicar el análisis de componentes conectados a la imagen con umbral
output = cv.connectedComponentsWithStats(dilatation2, connectivity, cv.CV_32S)
(numLabels, labels, stats, centroids) = output
# inicializar una máscara de salida para almacenar todos los insectos de la imagen
mask = np.zeros(dilatation2.shape, dtype="uint8")
(numLabels, labels, stats, centroids) = output
# bucle sobre el número de etiquetas de componentes conectados únicos
for i in range(1, numLabels):
	# si este es el primer componente entonces examinamos el
	# *fondo* (típicamente simplemente ignoraríamos estod
	# componente en nuestro ciclo)
	if i == 1:
		text = "examining component {}/{} (start)".format(
			i + 1, numLabels)
	# de lo contrario, estamos examinando un componente conectado real
	else:
		text = "examining component {}/{}".format( i + 1, numLabels)
	# print a status message update for the current connected
	# component
	print("[INFO] {}".format(text))
	# imprimir una actualización de mensaje de estado para el 
	# componente conectado actual
	x = stats[i, cv.CC_STAT_LEFT]
	y = stats[i, cv.CC_STAT_TOP]
	w = stats[i, cv.CC_STAT_WIDTH]
	h = stats[i, cv.CC_STAT_HEIGHT]
	area = stats[i, cv.CC_STAT_AREA]
	(cX, cY) = centroids[i]
	# clonar nuestra imagen original (para que podamos dibujar sobre ella) 
	# y luego dibujar un cuadro delimitador que rodee el componente conectado 
	# junto con un círculo correspondiente al centroide
	output = dilatation2.copy()
	cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
	cv.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
	# construir una máscara para el componente conectado actual 
	# encontrando píxeles en la matriz de etiquetas que tengan 
	# el ID del componente conectado actual
	componentMask = (labels == i).astype("uint8") * 255
	# show our output image and connected component mask
	cv.imshow("Output", output)
	cv.imshow("Connected Component", componentMask)
	# mostrar cada sección de imagen para cada insecto
	final_image = cv.bitwise_and(img, img, mask=componentMask)
	# recorta cada etiqueta con un margen
	cropped_final = final_image[y:y+h,x:x+w]
	# mostrar cada etiqueta correspondiente a cada insecto detectado
	cv.imshow("Final Image", cropped_final)
	
	input_path = r"Root/Prueba/*.png"

	# make sure below folder already exists
	out_path = "Root/Prueba/Plaga/"

	image_paths = list(glob.glob(input_path))
	for numLabels, imag in enumerate(image_paths):
		# cambiar el tamaño de la imagen en 100% para una mejor vista
		scale_percent = 100 # por ciento de la escala original
		width = int(cropped_final.shape[1] * scale_percent / 100)
		height = int(cropped_final.shape[0] * scale_percent / 100)
		dim = (width, height)
  
		# resize image
		final_image = cv.resize(cropped_final, dim, interpolation = cv.INTER_AREA)
		cv.imwrite(out_path + f'prueba5_{str(i)}.png', final_image)
		cv.waitKey(0)
cv.waitKey(0)


	
