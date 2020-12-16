# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
from PIL import Image
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Keras
import keras
from keras import models, optimizers
from keras import layers
from keras.models import Model
import warnings

'''layers.Conv2D(32, 3,activation='relu'),
            layers.BatchNormalization(axis=3),
            layers.Activation('relu'),
            layers.MaxPooling2D((5. 5)),

            layers.Conv2D(64, 3, activation='relu'),
            layers.BatchNormalization(axis=3),
            layers.Activation('relu'),
            layers.MaxPooling2D((5. 5)),

            layers.Conv2D(128, 3, activation='relu'),
            layers.BatchNormalization(axis=3),
            layers.Activation('relu'),
            layers.MaxPooling2D((5. 5)),
'''


class SpectroExpert():
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.model = None

    def build(self, input_shape, learning_rate=0.0005, classes=10):
        self.model = models.Sequential([
            layers.Conv2D(32, 5, activation='relu', input_shape=input_shape),
            layers.BatchNormalization(axis=3),
            layers.MaxPooling2D((5, 5)),

            layers.Conv2D(64, 5, activation='relu'),
            layers.BatchNormalization(axis=3),
            layers.MaxPooling2D((5, 5)),

            layers.Conv2D(64, 5, activation='relu'),
            layers.BatchNormalization(axis=3),
            layers.MaxPooling2D((5, 5)),

            layers.Flatten(),
            layers.Dropout(rate=0.5),
            layers.Dense(128, activation='relu', name='out_features'),

            layers.Dense(classes, activation='softmax',
                         name='fc' + str(classes))
        ], name='SpectroModel')
        self.opt = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=self.opt,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, x_train, y_train, **params):
        self.history = self.model.fit(x_train,
                                      y_train,
                                      **params)
        return self.history

    def fit_gen(self, train_gen, **params):
        self.history = self.model.fit_generator(train_gen,
                                                **params)
        return self.history

    def predict(self, x_test):
        return self.core_model.predict(x_test)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def build_core_model(self, learning_rate=0.0005):
        self.core_model = Model(inputs=self.model.input,
                                outputs=self.model.get_layer('out_features').output)
        self.core_model.compile(optimizer=self.opt,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        return self.core_model


# %%
