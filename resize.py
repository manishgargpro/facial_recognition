import csv
import os
import shutil
from PIL import Image
from pprint import pprint

labelpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\label\label.csv'
imagepath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\image\cropped'
destpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\image\resized'

res = 224

for d, e, f in os.walk(imagepath):
    for fi in f:
        im = Image.open(os.path.join(imagepath, fi))
        im = im.resize((res, res))
        im.save(os.path.join(destpath, fi))
