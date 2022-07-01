import os
from sqlite3 import Row
import matplotlib.pyplot

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

import tensorflow as tf

from tensorflow.python.keras.layers import Input, Dense
#from tensorflow import keras
import tensorflow_datasets as tfds

#If you use GPU this might save you some errors
#physical_devices=tf.config.list_phisical_divices("GPU")
#tf.config.experimenta.set_memory_growth(physical_devices[0],True)

(ds_train), ds_info = tfds.load(
    "citrus_leaves",
    split=["train"],
    shuffle_files=True,
    download = True,
    as_supervised=True,
    with_info=True,


)
print(ds_train)

#tfds.show_examples(ds_train, ds_info, rows=3, cols=3 )







