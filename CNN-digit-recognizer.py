# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
dirname = os.path.dirname(__file__)

# Any results you write to the current directory are saved as output.

train = pd.read_csv(dirname+"/data/train.csv")
test = pd.read_csv(dirname+"/data/test.csv")

Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)

del train

#print(X_train.shape)
#print(test.shape)

#X_train.isnull().sum().sum()
#test.isnull().sum().sum()

X_train = X_train / 255.
test = test / 255.

X_train = X_train.values.reshape([-1, 28, 28, 1])
test = test.values.reshape([-1, 28, 28, 1])

Y_train = keras.utils.np_utils.to_categorical(Y_train, num_classes=10)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=1)

strides = (2, 2)

lr_reduction = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 3, verbose = 1, factor = 0.5, min_lr = 0.00001)

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "Same", activation = "relu", input_shape = (28, 28, 1)))
model.add(keras.layers.MaxPool2D((2, 2), strides = strides))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = "Same", activation = "relu"))
model.add(keras.layers.MaxPool2D((2, 2), strides = strides))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = "Same", activation = "relu"))
model.add(keras.layers.MaxPool2D((2, 2), strides = strides))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation = "relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(10, activation = "softmax"))

model.summary()
optimizer = keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

datagen = ImageDataGenerator(rotation_range = 20, zoom_range = 0.2, width_shift_range = 0.2, height_shift_range = 0.1,
                            horizontal_flip = False, vertical_flip = False)

history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size = 64), steps_per_epoch = X_train.shape[0]//64, epochs = 20,
                              verbose = 2, validation_data = (X_val, Y_val), callbacks = [lr_reduction])

results = model.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)