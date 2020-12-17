# %%
import csv
import os
import pathlib
import warnings
import pickle

# Keras
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from keras import layers, models, optimizers
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from AudioExpert import AudioExpert
from SpectroExpert import SpectroExpert
from TerminalExpert import TerminalExpert
from SpectroGenerator import SpectroGenerator
from AudioGenerator import AudioGenerator


class Combiner():
    def __init__(self, prepare=False):
        self.data_dir = 'processed_data'
        self.spec_expert = SpectroExpert()
        self.audio_expert = AudioExpert()
        self.terminal_expert = TerminalExpert()
        if prepare:
            self.prepare_data()
        self.generate_spec_data()

    def prepare_data(self):
        spec_gen = SpectroGenerator()
        spec_gen.create_dirs()

        audio_gen = AudioGenerator()
        audio_gen.create_dirs()
        audio_gen.split_test_files()

        # Spectrogram generation
        spec_gen.generate_wav_segs()
        spec_gen.generate_spectrograms()

        # Audio feature generation
        audio_gen.generate_audio_csv()

    def generate_spec_data(self):
        train_dir = f'{self.data_dir}/train/audio_features/audio_features.csv'

        self.train_df = pd.read_csv(train_dir)

        train_datagen = ImageDataGenerator(rescale=1./255)
        img_size = (250, 400)
        self.input_shape = (*img_size, 4)
        self.batch_size = 64
        self.spec_train_gen = train_datagen.flow_from_dataframe(
            self.train_df, target_size=img_size, color_mode="rgba", class_mode='categorical',
            batch_size=self.batch_size, shuffle=False, x_col='filename_png', y_col='genre')

        test_dir = f'{self.data_dir}/test/audio_features/audio_features.csv'
        self.test_df = pd.read_csv(test_dir)

        test_datagen = ImageDataGenerator(rescale=1./255)
        self.spec_test_gen = test_datagen.flow_from_dataframe(
            self.test_df, target_size=img_size, color_mode='rgba',
            class_mode='categorical', batch_size=self.batch_size,
            shuffle=False, x_col='filename_png', y_col='genre')

        # Audio features
        self.train_dfx = self.train_df.drop(columns=['filename_wav', 'filename_png', 'genre', 'label'])
        self.train_dfx.drop(self.train_dfx.columns[0], axis=1, inplace=True)
        
        self.test_dfx = self.train_df.drop(columns=['filename_wav', 'filename_png', 'genre', 'label'])
        self.test_dfx.drop(self.test_dfx.columns[0], axis=1, inplace=True)

        self.audio_ft_train = self.batchize(self.train_dfx)
        self.audio_ft_test = self.batchize(self.test_dfx)

    def batchize(self, df):
        x = df.values
        for i in range(0, len(x), self.batch_size):
            yield x[i:i + self.batch_size]

        
    def build_experts(self):
        self.spec_expert.build(self.input_shape, classes=10)
        self.terminal_expert.build(self.input_shape, classes=10)

    def train_spectro_expert(self, epochs=70):
        self.spec_expert_hist = self.spec_expert.fit_gen(self.spec_train_gen, epochs=epochs,
                                 validation_data=self.spec_test_gen)
        self.spec_expert.build_core_model()
        return self.spec_expert_hist
    
    def predict(self, x_test):
        spec_features = self.spec_expert.predict(x_test)
        return spec_features

    def concat_spect_aud(self):
        batch_index = 0
        x_train = []
        y_train = []
        print("Starting concat_spect_aud...")
        while batch_index <= self.spec_train_gen.batch_index:
            x_spec, y_spec = self.spec_train_gen.next()
            x_aud = next(self.audio_ft_train)
            x_spec_ft = self.spec_expert.predict(x_spec)
            # print(np.array(x_spec).shape, np.array(x_aud).shape, np.array(x_spec_ft).shape)    
            x_spec_aud = np.concatenate((np.array(x_spec_ft), np.array(x_aud)), axis=1)
            x_train.append(np.array(x_spec_aud))
            y_train.append(np.array(y_spec))
            batch_index = batch_index + 1

        unbatched_x_tr = [sample.reshape(1, x_spec_aud.shape[1]) for batch in x_train for sample in batch]
        x_train = np.array(unbatched_x_tr).reshape(len(unbatched_x_tr), 154)

        unbatched_y_tr = [sample.reshape(1, y_spec.shape[1]) for batch in y_train for sample in batch]
        y_train = np.array(unbatched_y_tr).reshape(len(unbatched_y_tr), 10)

        # get test data from gens
        batch_index = 0
        x_test = []
        y_test = []
        while batch_index <= self.spec_test_gen.batch_index:
            x_spec, y_spec = self.spec_test_gen.next()
            x_aud = next(self.audio_ft_test)
            x_spec_ft = self.spec_expert.predict(x_spec)
            # print(np.array(x_spec).shape, np.array(x_aud).shape, np.array(x_spec_ft).shape)    
            x_spec_aud = np.concatenate((np.array(x_spec_ft), np.array(x_aud)), axis=1)
            x_test.append(np.array(x_spec_aud))
            y_test.append(np.array(y_spec))
            batch_index = batch_index + 1

        unbatched_x_test = [sample.reshape(1, x_spec_aud.shape[1]) for batch in x_test for sample in batch]
        x_test = np.array(unbatched_x_test).reshape(len(unbatched_x_test), 154)
        unbatched_y_test = [sample.reshape(1, y_spec.shape[1]) for batch in y_test for sample in batch]
        y_test = np.array(unbatched_y_test).reshape(len(unbatched_y_test), 10)
        y_tr_labels = self.train_df['genre'].values
        y_ts_labels = self.test_df['genre'].values
        print("Done with concat_spect_aud!")

        return x_train, x_test, y_train, y_test, y_tr_labels, y_ts_labels

    def train_terminal_expert(self, x_train, x_test, y_train, y_test, epochs=200):
        self.terminal_expert.build(input_shape=x_train[0].shape, classes=10)
        self.terminal_expert_hist = self.terminal_expert.fit(x_train, y_train,\
                batch_size=self.batch_size,\
                validation_data=(x_test, y_test),\
                epochs=epochs
                    )
        return self.terminal_expert_hist
    
    def save_model(self, obj, filename):
        with open(filename+'.pickle', 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    def load_model(self, filename):
        model = None
        with open(filename+'.pickle', 'rb') as output:  # Overwrites any existing file.
            model = pickle.load(output)
        return model

# %%
