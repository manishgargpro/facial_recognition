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

numnotequal = 0
numundersize = 0
numveryundersize = 0
numextremelyundersize = 0
numcanpad = 0

res = 224

for d, e, f in os.walk(imagepath):
    for fi in f:
        imagelist.append(fi)
        fim = Image.open(os.path.join(imagepath, fi))
        w, h = fim.size
        if w == res - 1:
            numcanpad += 1
        # if w < res:
        #     numundersize += 1
        #     if w < res/2:
        #         numveryundersize += 1
        #         if w < res/4:
        #             numextremelyundersize += 1

# print(numundersize, len(imagelist), numundersize/len(imagelist))
# print(numveryundersize, len(imagelist), numveryundersize/len(imagelist))
# print(numextremelyundersize, len(imagelist),
#       numextremelyundersize/len(imagelist))
print(numcanpad, len(imagelist), numcanpad/len(imagelist))

# im = Image.open(os.path.join(imagepath, imagelist[0]))
# im = im.resize((res, res))
# im.show()
