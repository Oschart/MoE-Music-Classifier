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

#Keras
import keras
from keras import models
from keras import layers

import warnings
class SuperClassifier():
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.model = None
        
    def build(self, input_shape, classes=10):
        self.model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=input_shape),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        
    def fit(self, x_train, y_train, **params):
        self.history = self.model.fit(x_train,
                    y_train,
                    **params)
        return self.history
    
    def predict(self, x_test):
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

