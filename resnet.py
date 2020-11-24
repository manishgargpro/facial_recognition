import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns
import tensorflow as tf
import pickle
from keras.models import Sequential, load_model
from keras.applications import (
    ResNet50,
    ResNet50V2,
    ResNet101,
    ResNet101V2,
    ResNet152,
    ResNet152V2,
    DenseNet121,
    DenseNet169,
    DenseNet201
)
# from keras.applications.vgg16 import (
#     VGG16,
#     preprocess_input
# )
from keras.applications.densenet import preprocess_input, decode_predictions
from keras.layers import Conv2D, BatchNormalization, Dropout
from keras.layers import AveragePooling2D
from keras.layers import Dense, Flatten, MaxPool2D
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
# for conversion to one-hot coding
from keras.utils import to_categorical, vis_utils, plot_model
from sklearn.metrics import confusion_matrix
# ======
# funtion for loading datasets:
# you have to make changes, if you are not using fashion_mnist dataset

imgtype = "full"

trainlabelpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\label\label_filtered.csv'
trainimgpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\image\{}_train'.format(
    imgtype)
testlabelpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\label\label_test.csv'
testimgpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\image\{}_test'.format(
    imgtype)

sizevgg16 = 64
sizedensenet = 224


def getimages(size):
    Xtrain = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory=trainimgpath,
        target_size=(size, size),
        interpolation='bilinear'
    )
    Xtest = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory=testimgpath,
        target_size=(size, size),
        interpolation='bilinear',
        shuffle=False
    )
    return Xtrain, Xtest


# def createvgg16():
#     vggfeats = VGG16(weights='imagenet', include_top=False,
#                      input_shape=(sizevgg16, sizevgg16, 3))
#     for layer in vggfeats.layers:
#         layer.trainable = False
#     model = Sequential()
#     model.add(vggfeats)
#     model.add(Flatten())
#     layernum = 2
#     units = 1024
#     for i in range(layernum):
#         model.add(Dense(units=units, activation="relu"))
#     model.add(Dense(units=7, activation="softmax"))
#     return model, layernum, units


def createdensenet():
    densenetfeats = DenseNet201(weights='imagenet', include_top=False,
                                input_shape=(sizedensenet, sizedensenet, 3))
    densenetfeats.summary()
    for layer in densenetfeats.layers:
        layer.trainable = False
    # densenetfeats.summary()
    model = Sequential()
    model.add(densenetfeats)
    model.add(Flatten())
    layernum = 4
    units = 1024
    for i in range(layernum):
        model.add(Dense(units=units, activation="relu"))
    model.add(Dense(units=7, activation="softmax"))
    return model, layernum, units


def saveaccplot(epochs, acc, val_acc, modelname):
    fig1, ax1 = plt.subplots()
    ax1.plot(epochs, acc, 'r', label="Training Accuracy")
    ax1.plot(epochs, val_acc, 'b', label="Validation Accuracy")
    ax1.set_xticks(epochs)
    ax1.set_ylim(bottom=0, top=1)
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    fig1.savefig("Acc_" + modelname + ".png")


def savelossplot(epochs, loss, val_loss, modelname):
    fig2, ax2 = plt.subplots()
    ax2.plot(epochs, loss, 'm', label="Training Loss")
    ax2.plot(epochs, val_loss, 'c', label="Validation Loss")
    ax2.set_xticks(epochs)
    ax2.set_ylim(bottom=0)
    ax2.set_title("Training and Validation Loss")
    ax2.legend()
    fig2.savefig("Loss_" + modelname + ".png")


def loadpretrained():
    # model, layernum, units = createvgg16()
    # Xtrain, Xtest = getimages(sizevgg16)
    model, layernum, units = createdensenet()
    Xtrain, Xtest = getimages(sizedensenet)
    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    observations = model.fit(Xtrain, epochs=20,
                             batch_size=8,
                             validation_data=(Xtest), verbose=1)
    acc = observations.history['accuracy']
    val_acc = observations.history['val_accuracy']
    loss = observations.history['loss']
    val_loss = observations.history['val_loss']
    epochs = range(1, len(acc) + 1)
    modelname = imgtype + "_" + model.layers[0].name + "_e" + str(len(acc)) + "_u" + str(
        layernum) + "by" + str(units) + "_res" + str(sizedensenet)
    model.save(modelname)
    saveaccplot(epochs, acc, val_acc, modelname)
    savelossplot(epochs, loss, val_loss, modelname)
    # model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # features = model.predict()
    # print(features.shape)
    # pickle.dump(features, open('test2.pkl', 'wb'))
    # model.summary()


def loadsaved(modelpath, imgpath):
    model = load_model(modelpath)
    # model.summary()
    # Xtest = getimages(sizedensenet)
    # XtestGroundTruth = Xtest.classes
    # pred = model.predict_classes(Xtest)
    # create_confusion_matrix(pred, XtestGroundTruth)
    # _, accr = model.evaluate(Xtest,  verbose=1)
    # print('Final validation accuracy: %.3f' % (accr*100.0))
    img = load_img(
        imgpath,
        target_size=(sizedensenet, sizedensenet),
        interpolation='bilinear'
    )
    img = img_to_array(img)
    # print(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2],))
    img = img/255
    # print(img)
    pred = model.predict(img)
    print("angry: " + str(pred[0][0]))
    print("disgust: " + str(pred[0][1]))
    print("fear: " + str(pred[0][2]))
    print("happy: " + str(pred[0][3]))
    print("sad: " + str(pred[0][4]))
    print("surprise: " + str(pred[0][5]))
    print("neutral: " + str(pred[0][6]))


def create_confusion_matrix(Ypredicted, YtestGroundTruth):
    classes = [0, 1, 2, 3, 4, 5, 6]
    y_pred = Ypredicted
    con_mat = tf.math.confusion_matrix(
        labels=YtestGroundTruth, predictions=y_pred)
    con_mat_norm1 = con_mat/con_mat.numpy().sum(axis=1)[:, tf.newaxis]
    con_mat_norm = np.around(con_mat_norm1, decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)
    #figure = plt.figure(figsize=(8, 8))
    plt.figure(figsize=(7, 7))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# loadpretrained()
loadsaved("full_densenet201_e20_u4by1024_res224",
          "genuine-smile-facial-expressions.jpeg.jpg")


# fig2, ax2 = plt.subplots()
# ax2.plot(range(1, 21), [
#     0.5710,
#     0.6208,
#     0.6420,
#     0.6638,
#     0.6813,
#     0.6958,
#     0.7134,
#     0.7284,
#     0.7424,
#     0.7564,
#     0.7714,
#     0.7815,
#     0.7928,
#     0.8024,
#     0.8116,
#     0.8211,
#     0.8304,
#     0.8381,
#     0.8442,
#     0.8505,
# ], 'r', label="Training Accuracy")
# ax2.plot(range(1, 21), [
#     0.6137,
#     0.6398,
#     0.6568,
#     0.6790,
#     0.7023,
#     0.7254,
#     0.7281,
#     0.7546,
#     0.7554,
#     0.7692,
#     0.7862,
#     0.8059,
#     0.8213,
#     0.8345,
#     0.8360,
#     0.8186,
#     0.8567,
#     0.8477,
#     0.8650,
#     0.8558,
# ], 'b', label="Validation Accuracy")
# ax2.set_xticks(range(1, 21))
# ax2.set_ylim(top=1, bottom=0)
# ax2.set_title("Training and Validation Accuracy")
# ax2.legend()
# fig2.savefig("Acc_cropped_densenet201_e20_u4by1024_res224.png")
