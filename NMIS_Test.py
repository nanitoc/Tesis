import os
from sqlite3 import Row


import matplotlib.pyplot



os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds

#If you use GPU this might save you some errors
#physical_devices=tf.config.list_phisical_divices("GPU")
#tf.config.experimenta.set_memory_growth(physical_devices[0],True)

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train","test"],
    shuffle_files=True,
    as_supervised=True,
    with_info= True,


)


#print(ds_train)
#fig = tfds.show_examples(ds_train, ds_info, rows=4, cols=4)


def normalize_img(image,lable):
    return tf.cast(image,tf.float32)/255.0,lable

AUTOTUNE= tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
ds_train = ds_train.map(normalize_img, num_parallel_calls = AUTOTUNE)
ds_train = ds_train.cache
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch (BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.prefetch(AUTOTUNE)


model = keras.Sequential([
    keras.input((28,28,1)),
    layers.Conv2D(323, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(10),


])

model.compile ( 
    optimizer=keras.optimizers.Adam(lr=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
 )



model.fit (ds_train,epochos=5,verbose=2)
model.evaluate(ds_test)


