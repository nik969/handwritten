import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras

keras.backend.set_image_data_format('channels_last')  # th

# Load model
model = keras.models.load_model('saved_models/emnist_letter_model.h5')
print(model.summary())

class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

imageList = []
fileNameList = []

INPUT_FOLDER_PATH = 'result/resized_images/'
dirFiles = os.listdir(INPUT_FOLDER_PATH)
dirFiles = sorted(dirFiles, key=lambda x: int(os.path.splitext(x)[0]))
for filename in dirFiles:
    img = cv2.imread(os.path.join(INPUT_FOLDER_PATH, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #imt = cv2.GaussianBlur(imt, (3, 3), 0)
    cv2.imwrite('result/final_img/' + str(filename), img)  # only for reference...
    img = img / 255
    imageList.append(img)
    fileNameList.append(filename)


imageList = np.array(imageList)
print(imageList.shape)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = model.predict(imageList)
for i in range(imageList.shape[0]):
    print(class_names[int(np.argmax(predictions[i]))-1])
