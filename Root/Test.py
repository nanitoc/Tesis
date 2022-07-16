import cv2
import tensorflow as tf
import keras

CATEGORIES = ["Alert" , "No Alert " ]   #Para Modelo 1
#CATEGORIES = ["Alert" , "No Alert", "No Alert 2" ]


def prepare(img_array):
    IMG_SIZE=300
    #img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

model = keras.models.load_model("Root/Alert-noAlert-64x3-CNN-CROPPED-Dia-Mosquitos-DAugmented.model")

result=[]

def Test(Images):
    #Mendieta trabaja    aqui

    for x in range(0,len(Images)):

        prediction = model.predict([prepare(Images[x])])
        print(CATEGORIES[int(prediction[0][0])])   # printear
        result.append(CATEGORIES[int(prediction[0][0])])

    return result


    #Hasta aqui


    """"
    for x in range(1,5):
    prediction = model.predict([prepare(f'Train/TEST_ALERT  ({x}).jpg')])
    print(CATEGORIES[int(prediction[0][0])])

    for x in range(1,6):
    prediction = model.predict([prepare(f'Train/TEST_NO_ALERT ({x}).jpg')])
    print(CATEGORIES[int(prediction[0][0])])
    #'Train/Alert_T1.jpg',
    """