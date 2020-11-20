import csv
import os
import shutil
import itertools
from PIL import Image
from pprint import pprint

labelpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\label\label.csv'
testpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\label\label_test.csv'
imagepath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\image\origin'
newpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\image\full_test'
labels = csv.DictReader(open(labelpath))
# names = []
# confidences = []
# classes = []

# for l in labels:
#     # names.append(l["image_name"])
#     if not in classes:
#         classes.append(l["expression_label"])

# for d, e, f in os.walk(imagepath):
#     for fi in f:
#         imagelist.append(fi)

# for i in imagelist:
#     if i not in names:
#         shutil.copy2(os.path.join(imagepath, i), os.path.join(newpath, i))

# nameseen = []


def cropandsave(l):
    # for l in labels:
    img = Image.open(os.path.join(imagepath, l["image_name"]))
    crop = img.crop((int(l["face_box_left"]),
                     int(l["face_box_top"]),
                     int(l["face_box_right"]),
                     int(l["face_box_bottom"])
                     ))
    newname = l["image_name"].rsplit(
        ".", 1)[0] + "_" + l["face_id_in_image"] + ".jpg"
    crop.save(os.path.join(newpath, l["expression_label"], newname))


def justsave(l):
    # for l in labels:
    img = Image.open(os.path.join(imagepath, l["image_name"]))
    newname = l["image_name"].rsplit(
        ".", 1)[0] + "_" + l["face_id_in_image"] + ".jpg"
    img.save(os.path.join(newpath, l["expression_label"], newname))


num = 0
w = csv.writer(open(testpath, "a", newline=""))
for i in labels:
    num += 1
    # # print(num)
    if num % 10 == 0:
        justsave(i)
        # print(i)
    # w.writerow([
    #     i["image_name"],
    #     i["face_id_in_image"],
    #     i["face_box_top"],
    #     i["face_box_left"],
    #     i["face_box_right"],
    #     i["face_box_bottom"],
    #     i["face_box_confidence"],
    #     i["expression_label"]
    # ])

# print(num)

# print(len(list(dict.fromkeys(nameseen))))
