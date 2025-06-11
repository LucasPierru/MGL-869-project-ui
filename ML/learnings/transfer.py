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

class TransfertLearning(Cifar10.Cifar10):
    IMAGE_GENERATOR_BATCH_SIZE = 32
    IMAGE_RESIZE = 224

    def preprocess(image, label):
        image = tf.image.resize(image, (TransfertLearning.IMAGE_RESIZE, TransfertLearning.IMAGE_RESIZE))
        image = preprocess_input(image)
        return image, label

    def resize_data_set(self, batch_size=IMAGE_GENERATOR_BATCH_SIZE):
        train_ds = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        train_ds = train_ds.map(self.preprocess).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test))
        val_ds = val_ds.map(self.preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds

    def compile_model(self):

        input_shape = (TransfertLearning.IMAGE_RESIZE, TransfertLearning.IMAGE_RESIZE, 3)
        base_model = tf.keras.applications.EfficientNetB0(include_top=False)
        base_model.trainable = False

        inputs = Input(shape=input_shape, name="input_layer")
        
        x = preprocess_input(inputs)
        x = base_model(x, training=False)
        x = GlobalAveragePooling2D(name="pooling_layer")(x)
        x = Dense(self.num_classes)(x)
        outputs = Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)
        self.model = tf.keras.Model(inputs, outputs)

        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return float(lr * tf.math.exp(-0.1))


    def train_model(self, epochs=20):
        learning_rate_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler)
        batch_sizes = [32, 64, 128]
        results = {}
        for batch_size in batch_sizes:
            (train_ds, val_ds) = self.resize_data_set(batch_size=batch_size)
            self.history = self.model.fit(train_ds,epochs=epochs, batch_size=batch_size, validation_data=val_ds, callbacks=[self.early_stopping_callback, learning_rate_scheduler_callback])
            results[batch_size] = self.history.history

        return results


if __name__ == "__main__":
    transfer_learning = TransfertLearning()
    transfer_learning.load_data_set()
    transfer_learning.display_data_set_img()
    transfer_learning.compile_model()
    transfer_learning.train_model()
    transfer_learning.save_model('../models/transfer_learning')
    transfer_learning.plot_history('accuracy', 'val_accuracy')
    transfer_learning.plot_history('loss', 'val_loss')

    (_, train_ds) = transfer_learning.resize_data_set()

    transfer_learning.display_confusion_matrix(train_ds)
    
    url = "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse2.mm.bing.net%2Fth%3Fid%3DOIP.iPdpNo-x8niQPxYv0rUl4AHaEK%26cb%3Diwc2%26pid%3DApi&f=1&ipt=139baa52f7bcebe0ad4eabc04836ec45e049083e330b3996f1319b8ee837023e&ipo=images"
    transfer_learning.prediction_on_image(url, transfer_learning.model, transfer_learning.IMAGE_RESIZE)