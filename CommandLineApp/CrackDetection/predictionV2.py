from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

batch_size = 32
IMG_HEIGHT = 227
IMG_WIDTH = 227

import os

test_dir = os.path.join(r'../TestingImagesSmall')
model = tf.keras.models.load_model('./ResultingModels/ModelFromImageProcessorV2.h5')

test_image_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.3,
    validation_split=0.2
)  # Generator for our training data)

test_image_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=test_dir,
                                                           shuffle=True,
                                                           color_mode="grayscale",
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary',
                                                           classes=["Positives", "Negatives"])





model.summary()

print(test_image_gen.filepaths)
print(test_image_gen.labels)
evaluations = model.evaluate(test_image_gen, verbose=1)
predictions = model.predict(test_image_gen, verbose=1)

print("Predictions", predictions)