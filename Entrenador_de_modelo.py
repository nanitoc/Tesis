from copyreg import pickle
import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from keras.callbacks_v1 import TensorBoard
import time


Name= "Alert-noAlert-64x3-CNN-CROPPED-Dia-Mosquitosr-DAugmented{}".format(int(time.time())) #nombre para referenciar modelo entrenado
tensorboard = TensorBoard(log_dir='.logs/{}'.format(Name)) # crea un archivo para ser analizado en tensorBoard

#Se cargan los archivos .pickles creados en el programa extractor de data
x = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

#Se normaliza la informacion de la imagen
x = x/255.0

# Primera capa convolucional
model = Sequential()
model.add(Conv2D(64,(3,3), input_shape = x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
# segunda capa convolucional
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())   
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics= ['accuracy']
                    )

model.fit(x, y, batch_size=32, epochs=10, callbacks=[tensorboard], validation_split=0.1)
#model.fit(x, y, batch_size=32, epochs=3,validation_split=0.1)

model.save('Alert-noAlert-64x3-CNN-CROPPED-Dia-Mosquitos-DAugmented.model')   #Se guarda el modelo entrenado en un archivo .model
