import os
import csv
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.applications import ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2
from keras.layers import Conv2D, BatchNormalization, Dropout
from keras.layers import AveragePooling2D
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
# for conversion to one-hot coding
from keras.utils import to_categorical
# ======
# funtion for loading datasets:
# you have to make changes, if you are not using fashion_mnist dataset


labelpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\label\label.csv'
imagepath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\image\resized_224'
# destpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\image\'
rows = csv.DictReader(open(labelpath))
names = []
confidences = []
labels = []
imagelist = []

for r in rows:
    names.append(r["image_name"])
    confidences.append(r["face_box_confidence"])
    labels.append(r["expression_label"])

def getimages():  
  for d, e, f in os.walk(imagepath):
      for fi in f:
          fiarr = np.load(os.path.join(imagepath, fi))
          imagelist.append(fiarr)

def loadmodel():
  model = Sequential()
  model.add(ResNet50())

for i in imagelist:
    print(i)
