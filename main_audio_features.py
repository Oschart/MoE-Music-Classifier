# %%
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
%matplotlib inline
import os
from PIL import Image
import pathlib
import csv
from keras import models
from keras import layers
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from AudioExpert import AudioExpert
#Keras
import keras

import warnings
warnings.filterwarnings('ignore')

GENERATE_AUDIO_FEATURES = True
header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
genres = ['blues', 'classical', 'country',  'disco',  'hiphop',  'jazz',  'metal',  'pop',  'reggae',  'rock']
AUDIO_FEATURES_PATH = 'GTZAN/audio_features.csv'
wav_dir = 'GTZAN/genres_original/'

# Get Audio Features
if GENERATE_AUDIO_FEATURES:
    a_exp = AudioExpert()
    a_exp.fit()
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open(AUDIO_FEATURES_PATH, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    for g in genres:
        for filename in os.listdir(wav_dir+g):
            songname = wav_dir+g+'/'+filename
            features = a_exp.predict(songname)
            file = open(AUDIO_FEATURES_PATH, 'a', newline='')
            features.insert(0, filename)
            features.append(g)
            with file:
                writer = csv.writer(file)
                writer.writerow(features)
# %%
data = pd.read_csv(AUDIO_FEATURES_PATH)
data.head()
data = data.drop(['filename'],axis=1)
genre_list = data.iloc[:, -1]
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
x_train_i, x_test, y_train_i, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(x_train_i, y_train_i, shuffle=True, test_size=0.2)

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train,
                    y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val))
results = model.evaluate(x_test, y_test)

# %%
