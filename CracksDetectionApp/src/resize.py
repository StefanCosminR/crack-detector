#!/usr/bin/python
import os

from PIL import Image

path = "images/set2/no_crack/test/"
output_path = "images_resized/set2/no_crack/test/"
dirs = os.listdir(path)


def resize():
    for item in dirs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            f, e = os.path.splitext(output_path + item)
            im_resize = im.resize((128, 128), Image.ANTIALIAS)
            im_resize.save(f + '.jpg', 'JPEG', quality=99)


resize()
