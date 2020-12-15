from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow import keras
import os
from emnist import extract_training_samples
from emnist import extract_test_samples

# K.set_image_dim_ordering('th')
keras.backend.set_image_data_format('channels_last')  # th

train_images, train_labels = extract_training_samples('letters')
test_images, test_labels = extract_test_samples('letters')

class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

train_images = train_images / 255


# Defining the model
def create_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(1, 28, 28), activation='relu'))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(27, activation='softmax'))  # 27 is number of classes

    return model


# MAIN ...
if os.path.exists('saved_models/emnist_letter_model.h5'):
    print('Model found')
    model = keras.models.load_model('saved_models/emnist_letter_model.h5')
    print(model.summary())
else:
    model = create_model()
    model.compile(optimizer='adam',  # adabound >> sgd > adam > rmsprop
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=30)
    model.save('saved_models/emnist_letter_model.h5')
    print("model saved")





# Final evaluation of the model
# scores = model.evaluate(test_images, test_labels, verbose=0)
# print("CNN Error: %.2f%%" % (100 - scores[1] * 100))