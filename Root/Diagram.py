import cv2
import tensorflow as tf
import keras
from keras.utils.vis_utils import plot_model
import pydot
CATEGORIES = ["Alert" , "No Alert " ]   #Para Modelo 1
#CATEGORIES = ["Alert" , "No Alert", "No Alert 2" ]




model = keras.models.load_model("Root/Alert-noAlert-64x3-CNN-CROPPED-Dia-Mosquitos-DAugmented.model")

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

