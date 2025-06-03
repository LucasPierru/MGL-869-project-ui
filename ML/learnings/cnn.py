import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization,Conv2D, MaxPooling2D, Input, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import urllib
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import confusion_matrix
import seaborn as sns
import cifar10 as Cifar10

class CNN(Cifar10.Cifar10):

    def compile_model(self):
        self.model = tf.keras.Sequential([

            tf.keras.layers.Conv2D(30, 3, activation="relu", input_shape=self.input_shape),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Conv2D(30, 3, activation="relu"),

            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Conv2D(30, 3, activation="relu"),

            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(self.num_classes, activation="softmax")
        ])

        self.model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

    def train_model(self):
        (train_data, valid_data) = self.data_augmentation()
        self.history = self.model.fit(train_data,epochs=20,batch_size=32,validation_data=valid_data, callbacks=[self.early_stopping_callback])

if __name__ == "__main__":
    cnn = CNN()
    cnn.load_data()
    cnn.display_data_set_img()
    cnn.compile_model()
    cnn.train_model()
    cnn.save_model('../models/cnn.keras')
    cnn.plot_history('accuracy', 'val_accuracy')
    cnn.plot_history('loss', 'val_loss')

    cnn.display_confusion_matrix(train_ds)
