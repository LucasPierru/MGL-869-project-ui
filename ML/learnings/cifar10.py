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


class Cifar10:

    PATIENCE = 3

    def __init__(self):
        self.num_classes = 10
        self.input_shape = (32, 32, 3)
        self.class_names = ['airplane','automobile','bird','cat','deer', 'dog','frog','horse','ship','truck']
        self.batch_size = 64
        self.epochs = 20
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def early_stopping(self):
        return tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=Cifar10.PATIENCE)

    def display_data_set_img(self):
        fig = plt.figure(figsize=(10,5))
        for i in range(self.num_classes):
            ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
            idx = np.where(self.y_train[:]==i)[0]
            features_idx = self.x_train[idx,::]
            img_num = np.random.randint(features_idx.shape[0])
            im = (features_idx[img_num,::])
            ax.set_title(self.class_names[i])
            plt.imshow(im)
        plt.show()

    def load_data_set(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        print('x_train shape:', self.x_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')

    def data_augmentation(self):

        IMAGE_GENERATOR_SEED = 42
        IMAGE_GENERATOR_BATCH_SIZE = 32

        tf.random.set_seed(IMAGE_GENERATOR_SEED)

        train_datagen = ImageDataGenerator(rescale=1./255)
        valid_datagen = ImageDataGenerator(rescale=1./255)

        train_data = train_datagen.flow(x=self.x_train,y=self.y_train,batch_size=IMAGE_GENERATOR_BATCH_SIZE,seed=IMAGE_GENERATOR_SEED)
        valid_data = valid_datagen.flow(x=self.x_test,y=self.y_test,batch_size=IMAGE_GENERATOR_BATCH_SIZE,seed=IMAGE_GENERATOR_SEED)

        return train_data, valid_data
    
    
    def prediction_on_image(self,image_url, size):

        resp = urllib.request.urlopen(image_url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (size,size))

        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.show()

        image = image.reshape((1, size, size, 3))

        prediction = self.model.predict(image)
        
        print('Predicted class: ', self.class_names[prediction.argmax()])

    def save_model(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model.save(os.path.join(model_path, 'transfer_learning_model.keras'))
        print(f'Model saved to {model_path}')

    def plot_history(self, key, val_key):

        plt.figure(figsize=(15,6))

        plt.subplot(1,2,1)
        plt.plot(self.history.history[key], label='Train '+key.capitalize(), color='#8502d1')
        plt.plot(self.history.history[val_key], label='Validation '+key.capitalize(), color='darkorange')
        plt.legend()
        plt.title(key.capitalize()+' Evolution')

        plt.show()

    def display_confusion_matrix(self, validation_data=None):
            
        y_pred_probs = self.model.predict(validation_data)
        y_pred = tf.argmax(y_pred_probs, axis=1).numpy()

        y_true = []
        for _, labels in validation_data:
            y_true.extend(labels.numpy())
        y_true = np.array(y_true)

        conf_mat = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()