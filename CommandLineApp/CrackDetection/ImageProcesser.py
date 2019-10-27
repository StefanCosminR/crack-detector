import cv2
import os
import random
import pickle
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

images = list()
labels = list()

dirname = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(dirname, 'Images')

# incarcam imaginile in memorie

images_path = sorted(list(paths.list_images(data_path)))
random.seed(42)
random.shuffle(images_path)


for path in images_path:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (64, 64))
        # mai sus se poate face .flatten() la imagini

        images.append(image)
        label = path.split(os.path.sep)[-2]
        labels.append(label)

# separam imaginile in training set si testing set
# trainX si testX sunt imaginile, trainY si testY sunt labelurile

trainX, testX, trainY, testY = train_test_split(images, labels, test_size=0.25, train_size=0.75, random_state=42)

# transformare labeluri in "one-hot encoding"

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))


INIT_LR = 0.01
EPOCHS = 75


print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

"""

model_path = os.path.join(dirname, 'Model/crack.model')
label_path = os.path.join(dirname, 'Model/crack.pickle')

print("[INFO] serializing network and label binarizer...")
model.save(model_path)
f = open(label_path, "wb")
f.write(pickle.dumps(lb))
f.close()

"""