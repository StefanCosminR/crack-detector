from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import os
import numpy as np
import matplotlib.pyplot as plt
import aspectlib
import re
from datetime import datetime


batch_size = 32
IMG_HEIGHT = 227
IMG_WIDTH = 227


@aspectlib.Aspect
def log_args(*args, **kwargs):
    print("[aspect] Calling fn with args", args)
    result = yield aspectlib.Proceed
    yield aspectlib.Return(result)


@aspectlib.Aspect
def log_kwargs(*args, **kwargs):
    print("[aspect] Calling fn with kwargs", kwargs)
    result = yield aspectlib.Proceed
    yield aspectlib.Return(result)


@aspectlib.Aspect
def ensure_folder_exists(*args, **kwargs):
    path = args[1]
    parent_path = os.path.dirname(path)
    print('[aspect] Checking path for existance', parent_path)

    if not os.path.exists(parent_path):
        os.makedirs(parent_path)

    result = yield aspectlib.Proceed
    yield aspectlib.Return(result)

write_log_kwargs_called = False

@aspectlib.Aspect
def write_log_kwargs(*args, **kwargs):
    global write_log_kwargs_called
    path = './Logs/log.txt'
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'a') as file:
        if not write_log_kwargs_called:
            file.write('\n-----------------------------------------------------\n\nStarted: ' + str(datetime.now()) + "\n\n")
            write_log_kwargs_called = True
        file.write(str(kwargs) + "\n")

    result = yield aspectlib.Proceed
    yield aspectlib.Return(result)


def get_train_images_path():
    dirname = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(dirname, 'Images')


train_dir = get_train_images_path()


@log_kwargs
@write_log_kwargs
def get_image_generator(**kwargs):
    return keras.preprocessing.image.ImageDataGenerator(**kwargs)  # Generator for our training data)


@log_kwargs
@write_log_kwargs
def read_from_directory_using_generator(generator, **kwargs):
    return generator.flow_from_directory(**kwargs)

train_image_generator = get_image_generator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True,
        zoom_range=0.3,
        validation_split=0.2)


train_data_gen = read_from_directory_using_generator(train_image_generator,
                                                     batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     color_mode="grayscale",
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary',
                                                     classes=["Positives", "Negatives"])


# augmented_images = [train_data_gen[0][0][0] for i in range(5)]


# sample_training_images, _ = next(train_data_gen)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# plotImages(augmented_images)
# plotImages(sample_training_images[:5])

def createModel():
    model = Sequential([
        Conv2D(16, 1, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 1, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 1, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax'),
    ])

    return model


def compileModel(model):
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

def printModelSummary(model):
    model.summary()


# model.fit(
#     train_data_gen,
#     steps_per_epoch= 40000 // batch_size,
#     epochs=10
# )

def trainModel(model, train_data):
    history = model.fit_generator(
        train_data,
        epochs=1
    )

    return history




aspectlib.weave(plotImages, log_args)
aspectlib.weave(createModel, log_args)

model = createModel()
compileModel(model)
printModelSummary(model)

aspectlib.weave(model.save, ensure_folder_exists)


history = trainModel(model, train_data_gen)


model.save('./ResultingModels2/ModelFromImageProcessorV2.h5')

acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

loss = history.history['loss']
# val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
