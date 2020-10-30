import csv
import os
import shutil
from PIL import Image
from pprint import pprint

labelpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\label\label.csv'
imagepath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\image\origin'
newpath = r'C:\Users\Administrator\Desktop\Coding_stuff\facial_recognition\data\image\cropped'
labels = csv.DictReader(open(labelpath))
names = []
imagelist = []

# for l in labels:
#     names.append(l["image_name"])

# for d, e, f in os.walk(imagepath):
#     for fi in f:
#         imagelist.append(fi)

# for i in imagelist:
#     if i not in names:
#         shutil.copy2(os.path.join(imagepath, i), os.path.join(newpath, i))

nameseen = []
for l in labels:
    img = Image.open(os.path.join(imagepath, l["image_name"]))
    crop = img.crop((int(l["face_box_left"]),
                     int(l["face_box_top"]),
                     int(l["face_box_right"]),
                     int(l["face_box_bottom"])
                     ))
    newname = l["image_name"].rsplit(
        ".", 1)[0] + "_" + l["face_id_in_image"] + ".jpg"
    # print(newname)
    nameseen.append(newname)
    crop.save(os.path.join(newpath, newname))

print(len(list(dict.fromkeys(nameseen))))
