import csv
import os
import shutil
from PIL import Image
from pprint import pprint

labelpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\label\label.csv'
imagepath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\image\cropped'
rows = csv.DictReader(open(labelpath))
names = []
confidences = []
labels = []
imagelist = []


for r in rows:
    names.append(r["image_name"])
    confidences.append(r["face_box_confidence"])
    labels.append(r["expression_label"])


for d, e, f in os.walk(imagepath):
    for fi in f:
        imagelist.append(fi)


res = 224

im = Image.open(os.path.join(imagepath, imagelist[0]))
im = im.resize((res, res))
im.show()
