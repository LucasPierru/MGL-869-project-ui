from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPool2D,Conv2D, Dense, Dropout, Activation, Flatten, BatchNormalization,Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import cv2
import urllib
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cifar10 as Cifar10
from keras.callbacks import LearningRateScheduler, EarlyStopping


class CNN(Cifar10.Cifar10):
    KERNEL_SIZE = (3, 3)

    def compile_model(self):
        model = Sequential()

        # Convolutional Layer
        model.add(Conv2D(filters=32, kernel_size=CNN.KERNEL_SIZE, input_shape=self.input_shape ,activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=32, kernel_size=CNN.KERNEL_SIZE, input_shape=self.input_shape ,activation='relu', padding='same'))
        model.add(BatchNormalization())
        # Pooling layer
        model.add(MaxPool2D(pool_size=(2, 2)))
        # Dropout layers
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=64, kernel_size=CNN.KERNEL_SIZE, input_shape=self.input_shape, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=CNN.KERNEL_SIZE, input_shape=self.input_shape, activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=128, kernel_size=CNN.KERNEL_SIZE, input_shape=self.input_shape ,activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=CNN.KERNEL_SIZE, input_shape=self.input_shape ,activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        # model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(10, activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return float(lr * tf.math.exp(-0.1))

    def train_model(self):

        callbacks = [
            EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True),
            LearningRateScheduler(self.lr_scheduler)
        ]

        (train_data, valid_data) = self.data_augmentation()

        self.history = self.model.fit(
            train_data,
            epochs=50,
            batch_size=64,
            validation_data=valid_data,
            callbacks=callbacks
        )

if __name__ == "__main__":
    cnn = CNN()
    cnn.load_data()
    cnn.display_data_set_img()
    cnn.compile_model()
    cnn.train_model()
    cnn.save_model('../models/cnn.keras')
    cnn.plot_history('accuracy', 'val_accuracy')
    cnn.plot_history('loss', 'val_loss')

    (train_ds, _) = cnn.data_augmentation()

    cnn.display_confusion_matrix(train_ds)
