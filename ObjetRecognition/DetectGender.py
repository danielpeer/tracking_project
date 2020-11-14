import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop, adam
from keras.preprocessing import image

current_dir = "D:\\Users\\97252\PycharmProjects\\tracking_project\\ObjetRecognition"


class GenderDetector:

    def __init__(self):
        train = ImageDataGenerator(rescale=1 / 255)
        validation = ImageDataGenerator(rescale=1 / 255)

        train_dataset = train.flow_from_directory(os.path.join(current_dir, 'train'),
                                                  target_size=(100, 100),
                                                  batch_size=1,
                                                  class_mode='binary'
                                                  )
        validation_dataset = validation.flow_from_directory(os.path.join(current_dir, 'train'),
                                                            target_size=(100, 100),
                                                            batch_size=1,
                                                            class_mode='binary'
                                                            )

        model = Sequential()
        print(train_dataset.class_indices)

        model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))

        model.summary()

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.summary()

        opt = RMSprop(lr=0.001)

        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        model.fit_generator(train_dataset,
                            steps_per_epoch=70,
                            epochs=10,
                            validation_data=validation_dataset,
                            validation_steps=5)

        self.model = model

    def get_predict(self, img):
        cv2.imwrite(os.path.join(current_dir, "a.jpg"), img)
        img_pred = image.load_img(os.path.join(current_dir, "a.jpg"), target_size=(100, 100))
        img_pred = image.img_to_array(img_pred)
        img_pred = np.expand_dims(img_pred, axis=0)
        if self.model.predict_classes(img_pred)[[0]] == 0:
            return "female"
        else:
            return "male"