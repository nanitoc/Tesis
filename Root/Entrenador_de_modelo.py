from copyreg import pickle
import pickle
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from keras.callbacks_v1 import TensorBoard
import time


Name= "V35_Alert-noAlert-64x3-CNN-CROPPED-Dia-Mosquitos-DAugmented{}".format(int(time.time())) #nombre para referenciar modelo entrenado
tensorboard = TensorBoard(log_dir='.logs/{}'.format(Name)) # crea un archivo para ser analizado en tensorBoard

# se cargan los archivos .pickles creados en el programa extractor de data
x = pickle.load(open("Root/x5.pickle","rb"))
y = pickle.load(open("Root/y5.pickle","rb"))

# se normaliza la informacion de la imagen
x = x/255.0

# # Primera capa convolucional
model = Sequential()
# segunda capa convolucional
model.add(Conv2D(8,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
# tercera capa convolucional
model.add(Conv2D(16,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
# cuarta capa convolucional
model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
# quinta capa convolucional
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

model.fit(x, y, batch_size=32, epochs=7, callbacks=[tensorboard], validation_split=0.05)
#model.fit(x, y, batch_size=32, epochs=3,validation_split=0.1)

model.save('Root/V35_Alert-noAlert-64x3-CNN-CROPPED-Dia-Mosquitos-DAugmented.model')   #Se guarda el modelo entrenado en un archivo .model
