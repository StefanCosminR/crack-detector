import cv2
import os
import random
import pickle
import numpy
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical

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
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (32, 32)).flatten()
        # mai sus se poate face .flatten() la imagini

        images.append(image)
        label = path.split(os.path.sep)[-2]
        labels.append(label)

"""
for i in range(0, len(labels)):
        if labels[i] == 'Positives':
                labels[i] = 1
        else:
                labels[i] = 0
"""

# scalam intensitatilor pixelilor ca sa fie intre [0, 1]

images = numpy.array(images, dtype="float") / 255.0
labels = numpy.array(labels)

#labels = labels.reshape(1, -1)

# separam imaginile in training set si testing set
# trainX si testX sunt imaginile, trainY si testY sunt labelurile

print(len(labels), len(images))


trainX, testX, trainY, testY = train_test_split(images, labels, test_size=0.25, train_size=0.75, random_state=42)

# transformare labeluri in "one-hot encoding"


#trainY = to_categorical(trainY)
#testY = to_categorical(testY)


# lb = LabelBinarizer()
# trainY = lb.fit_transform(trainY)
# testY = lb.transform(testY)

encoder = LabelEncoder()
encoder.fit(trainY)
trainY = encoder.transform(trainY)

encoder.fit(testY)
testY = encoder.transform(testY)


model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(encoder.classes_), activation="softmax"))


INIT_LR = 0.01
EPOCHS = 5


print(testY.argmax())

print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
train_acc = model.evaluate(trainX, trainY, verbose=0)
test_acc = model.evaluate(testX, testY, verbose=0)
print(train_acc, test_acc)
#print(classification_report(testY, predictions, target_names=encoder.classes_))


model_path = os.path.join(dirname, 'Model/crack.model')
label_path = os.path.join(dirname, 'Model/crack.pickle')

print("[INFO] serializing network and label binarizer...")
model.save(model_path)
f = open(label_path, "wb")
f.write(pickle.dumps(encoder))
f.close()

