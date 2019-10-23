import cv2
import os
from imutils import paths

images = list()
labels = list()

data_path = r'C:\Users\Serban\Desktop\Licenta\LicentaResources\animals'

images_path = sorted(list(paths.list_images(data_path)))

for path in images_path:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (64, 64))
        images.append(image)
        label = path.split(os.path.sep)[-2]
        labels.append(label)

cv2.imshow('image',images[1])

print(images[1])

cv2.waitKey(0)
