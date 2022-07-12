#this is for training DATA
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 
import random
import pickle


DATADIR= "C:/Users/jerem/Desktop/DATASET_ALERT-NOALERT(CROPPED)"    #Direccion de la base de datos
CATEGORIES = ["Alert" , "No Alert"]  #Labels de la base de datos

IMG_SIZE = 300          #Tamañ de la imagen



Training_data = []

#Función que lee y transforma cada imagen en la base de datos
def create_training_data():
    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        path = os.path.join(DATADIR, category)   
        for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    Training_data.append([new_array, class_num])
                except Exception as e:
                    pass


create_training_data()  # se llama la función

print(len(Training_data))   # se imprime el total de datos procesados



random.shuffle(Training_data)
#for sample in Training_data:
    #print(sample[1])

x = []      #variable utilizada para almacenar las caracteristicas de las imagenes
y = []      #variable utilizada para los labels


for features, label in Training_data:
    x.append(features) #Se agrega las caracterizticas a la lista X
    y.append(label)     #Se agrega los labels a la lista y

#se realiza un arreglo con Numpy
x = np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,1)     #En caso de querer trabajar en RBG cambiar el ultimo 1 a 3
y = np.array(y)
#Para Guardar tu data entrenada

#Se utiliza la libreria picle para la creacion de dos archivos .pickle que seran utilizados en el programa de entrenamiento
pickle_out = open ("X.pickle","wb")
pickle.dump(x,pickle_out)
pickle_out.close()

pickle_out = open ("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()
