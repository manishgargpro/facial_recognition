import os
import csv
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
import pickle
from keras.models import Sequential
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras.applications.vgg16 import (
    VGG16,
    preprocess_input
)
from keras.layers import Conv2D, BatchNormalization, Dropout
from keras.layers import AveragePooling2D
from keras.layers import Dense, Flatten, MaxPool2D
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
# for conversion to one-hot coding
from keras.utils import to_categorical
# ======
# funtion for loading datasets:
# you have to make changes, if you are not using fashion_mnist dataset


trainlabelpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\label\label_filtered.csv'
trainimgpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\image\cropped_train'
testlabelpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\label\label_test.csv'
testimgpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\image\cropped_test'
# rows = csv.DictReader(open(labelpath))
# names = []
# confidences = []
# labels = []
# imagelist = []

# for r in rows:
#     names.append(r["image_name"])
#     confidences.append(r["face_box_confidence"])
#     labels.append(r["expression_label"])


def getimages():
    Xtrain = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory=trainimgpath,
        target_size=(224, 224),
        interpolation='bilinear'
    )
    Xtest = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory=testimgpath,
        target_size=(224, 224),
        interpolation='bilinear'
    )
    # Ytrain = pd.read_csv(trainlabelpath, usecols=['expression_label'])
    # Ytest = pd.read_csv(testlabelpath, usecols=['expression_label'])
    return Xtrain, Xtest


def createvgg16():
    # model = Sequential()
    # model.add(Conv2D(input_shape=(224, 224, 3), filters=64,
    #                  kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=64, kernel_size=(3, 3),
    #                  padding="same", activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(filters=128, kernel_size=(
    #     3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=128, kernel_size=(
    #     3, 3), padding="same", activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(filters=256, kernel_size=(
    #     3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=256, kernel_size=(
    #     3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=256, kernel_size=(
    #     3, 3), padding="same", activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(filters=512, kernel_size=(
    #     3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=512, kernel_size=(
    #     3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=512, kernel_size=(
    #     3, 3), padding="same", activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(filters=512, kernel_size=(
    #     3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=512, kernel_size=(
    #     3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=512, kernel_size=(
    #     3, 3), padding="same", activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    vggfeats = VGG16(weights='imagenet', include_top=False,
                     input_shape=(224, 224, 3))
    for layer in vggfeats.layers:
        layer.trainable = False
    model = Sequential()
    model.add(vggfeats)
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=7, activation="softmax"))
    return model


def loadpretrained():
    model = createvgg16()
    Xtrain, Xtest = getimages()
    # for l in model.layers:
    #     print(l.output)
    adam = Adam(learning_rate=0.01)
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    observations = model.fit(Xtrain, epochs=2,
                             batch_size=1,
                             validation_data=(Xtest), verbose=1)
    acc = observations.history['accuracy']
    val_acc = observations.history['val_accuracy']
    loss = observations.history['loss']
    val_loss = observations.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label="Training Accuracy")
    plt.plot(epochs, loss, 'b', label="Training Loss")
    plt.title("Training Accuracy and Loss")
    plt.legend()
    plt.figure()
    plt.savefig("Training_Accuracy_and_Loss.png")
    plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
    plt.title("Validation Accuracy and Loss")
    plt.legend()
    plt.figure()
    plt.savefig("Validation_Accuracy_and_Loss.png")
    # _, acc = model.evaluate(Xtest,  verbose=1)
    # print('Final validation accuracy: %.3f' % (acc*100.0))
    # model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # features = model.predict()
    # print(features.shape)
    # pickle.dump(features, open('test2.pkl', 'wb'))
    # model.summary()


loadpretrained()
# getimages()
