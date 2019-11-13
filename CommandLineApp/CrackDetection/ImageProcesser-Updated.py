import cv2
import os
import random
import pickle
import numpy
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import ImageProcesserArhitecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


images = list()
labels = list()

dirname = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(dirname, 'Images')    #fara un s la final

if os.path.isdir(data_path) is False:
        raise Exception("Folder not found")

image_width = 32
image_height = 32

if (image_width is not 32) and (image_width is not 64):
        raise Exception("Invalid image width")

if (image_height is not 32) and (image_height is not 64):
        raise Exception("Invalid image height")

# incarcam imaginile in memorie

images_path = sorted(list(paths.list_images(data_path)))
random.seed(42)
random.shuffle(images_path)


for path in images_path:
        try:
                image = cv2.imread(path)
        except cv2.error as e:
                print(str(e))
                raise
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        try:
                image = cv2.resize(image, (32, 32)).flatten()
        except cv2.error as e:
                print(str(e))
                raise

        # mai sus se poate face .flatten() la imagini

        images.append(image)
        label = path.split(os.path.sep)[-2]
        labels.append(label)

# scalam intensitatilor pixelilor ca sa fie intre [0, 1]

images = numpy.array(images, dtype="float") / 255.0
labels = numpy.array(labels)

# separam imaginile in training set si testing set
# trainX si testX sunt imaginile, trainY si testY sunt labelurile

trainX, testX, trainY, testY = train_test_split(images, labels, test_size=0.25, train_size=0.75, random_state=42)

# transformare labeluri in "one-hot encoding"

le = LabelEncoder()
trainY = le.fit_transform(trainY)
trainY = to_categorical(trainY)

testY = le.fit_transform(testY)
testY = to_categorical(testY)

#model = Sequential()
#model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
#model.add(Dense(512, activation="sigmoid"))
#model.add(Dense(len(le.classes_), activation="softmax"))

model = ImageProcesserArhitecture.build_model(image_height, image_width, 3, classes=len(le.classes_))

print(le.classes_)

INIT_LR = 0.01
EPOCHS = 5

print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
train_acc = model.evaluate(trainX, trainY, verbose=0)
test_acc = model.evaluate(testX, testY, verbose=0)

print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))


model_path = os.path.join(dirname, 'Model/crack.h5')
label_path = os.path.join(dirname, 'Model/crack.pickle')

print("[INFO] serializing network and label binarizer...")
model.save(model_path)

try:
        f = open(label_path, "wb")
        f.write(pickle.dumps(le))
        f.close()
except Exception as e:
        print(str(e))
        raise


