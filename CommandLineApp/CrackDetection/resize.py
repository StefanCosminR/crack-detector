#!/usr/bin/python
import os

from PIL import Image

path = "../../CracksDetectionApp/images"
output_path = "images_resized"
dirs = os.listdir(path)


def resize():
    for item in dirs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            f, e = os.path.splitext(output_path + item)
            im_resize = im.resize((128, 128), Image.ANTIALIAS)
            im_resize.save(f + '.jpg', 'JPEG', quality=99)


resize()
