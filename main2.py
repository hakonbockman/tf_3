import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras

''' LOAD THE DATA AND SPLIT '''
data_dir_IR = pathlib.Path('./Combined/IR/')
data_dir_Visual = pathlib.Path('./Combined/Visual/')


batch_size = 32
img_height = 300
img_width = 300

''' TRAINING DATA'''
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_IR,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

''' VALIDATION DATA'''
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir_IR,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

print('class_names: ', class_names)

plt.figure(figsize=(15, 15))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()

#''' Alternative method '''
#'''
#import tensorflow_datasets as tfds
#
#
#tensorflow_datasets.disable_progress_bar()
#
#train_ds, validation_ds, test_ds = tensorflow_datasets.load(
#  
#)
#'''




''' Normalization of images '''
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

image_batch, labels_batch = next(iter(normalized_ds))

first_image = image_batch[0]

print(np.min(first_image), np.max(first_image))


''' CONFIGURE DATASET PERFORMANCE '''

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)



''' Compile the model'''

batch_size = 32
img_height = 300
img_width = 300

num_classes = train_ds.class_names

# Load in the Xception network pretrained on the imagenet dataset.
base_model = keras.applications.Xception(
  weights='imagenet',
  input_shape=(img_width, img_height, 3),
  include_top=False
)

base_model = False # Freeze the model, so we dont dissrupt the allready trained weights.

inputs = keras.Input(shape=(img_width, img_height, 3))

x = base_model(inputs, training = False)

outputs = keras.layers.Dense(1)(x)

model = keras.Model(inputs, outputs)

model.compile(
  optimizer = keras.optimizers.Adam(),
  loss = keras.losses.BinaryCrossentropy(from_logits=True),
  metrics = [keras.memtrics.BinaryAccuracy()]
)

