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
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder

from AudioExpert import AudioExpert
from SpectroExpert import SpectroExpert
from TerminalExpert import TerminalExpert
from SpectroGenerator import SpectroGenerator
from AudioGenerator import AudioGenerator
from SuperClassifier import SuperClassifier

class Combiner():
    def __init__(self, prepare=False):
        self.data_dir = 'processed_data'
        self.spec_expert = SpectroExpert()
        self.audio_expert = AudioExpert()
        self.terminal_expert = TerminalExpert()
        self.super_classifier = SuperClassifier()
        self.sub_classifierJCB = SuperClassifier()
        self.sub_classifierRH = SuperClassifier()
        self.sub_classifierRC = SuperClassifier()
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
        
        self.y_train = y_train
        self.y_test = y_test

        return x_train, x_test, y_train, y_test, y_tr_labels, y_ts_labels

    def get_audio_ft(self):
        self.enc = OneHotEncoder()
        self.scaler = StandardScaler()

        labels_tr = np.array(self.train_df['genre']).reshape(-1, 1)
        labels_ts = np.array(self.test_df['genre']).reshape(-1, 1)

        x_train = np.array(self.train_dfx.values, dtype=float)
        x_train = self.scaler.fit_transform(x_train)
        y_train = self.enc.fit_transform(labels_tr).toarray()

        x_test = np.array(self.test_dfx.values, dtype=float)
        x_test = self.scaler.transform(x_test)
        y_test = self.enc.transform(labels_ts).toarray()

        return x_train, x_test, y_train, y_test, labels_tr, labels_ts

    def train_terminal_expert(self, x_train, x_test, y_train, y_test, epochs=200):
        self.terminal_expert.build(input_shape=x_train[0].shape, classes=10)
        self.terminal_expert_hist = self.terminal_expert.fit(x_train, y_train,\
                batch_size=self.batch_size,\
                validation_data=(x_test, y_test),\
                epochs=epochs
                    )
        test_loss, test_acc = self.terminal_expert.evaluate(x_test, y_test)
        return self.terminal_expert_hist, test_loss, test_acc

    def train_audio_expert(self, x_train, x_test, y_train, y_test, epochs=200):
        self.audio_expert.build(input_shape=(x_train.shape[1],), classes=10)
        self.audio_expert_hist = self.audio_expert.fit(x_train, y_train,\
                batch_size=self.batch_size,\
                validation_data=(x_test, y_test),
                epochs=epochs
                )
        test_loss, test_acc = self.audio_expert.evaluate(x_test, y_test)
        return self.audio_expert_hist, test_loss, test_acc

    def train_superclassifier(self, mapping, concat=False, epochs=30):
        if concat:
            x_train, x_test, y_train, y_test, _, _ = self.concat_spect_aud()
        else:
            x_train, x_test, y_train, y_test, _, _ = self.get_audio_ft()

        y_tr_labels_super = []
        y_ts_labels_super = []
        for l in list(self.train_df['genre']):
            y_tr_labels_super.append(mapping[l])
        for l in list(self.test_df['genre']):
            y_ts_labels_super.append(mapping[l])
        self.super_enc = OneHotEncoder()
        y_train = self.super_enc.fit_transform(np.array(y_tr_labels_super).reshape(-1, 1)).toarray()
        y_test = self.super_enc.fit_transform(np.array(y_ts_labels_super).reshape(-1, 1)).toarray()
        self.super_classifier.build(input_shape=(x_train.shape[1],), classes=len(set(mapping.values())))
        self.super_classifier_hist = self.super_classifier.fit(x_train, y_train,\
                batch_size=self.batch_size,\
                validation_data=(x_test, y_test),
                epochs=epochs
                )
        test_loss, test_acc = self.super_classifier.evaluate(x_test, y_test)
        return self.super_classifier_hist, test_loss, test_acc

    def train_subclassifiers(self, concat=False, epochs=200):
        if concat:
            x_train, x_test, y_train, y_test, _, _ = self.concat_spect_aud()
        else:
            x_train, x_test, y_train, y_test, _, _ = self.get_audio_ft()

        x_train_JCB, y_train_JCB = [], []
        x_train_RH, y_train_RH = [], []
        x_train_RC, y_train_RC = [], []
        i = 0
        # there are no metal, disco or pop experts since it is a single-valued ccategory
        for (x, y) in zip(x_train, y_train):
            y_pred = self.super_classifier.predict(x.reshape(1,-1))
            super_category = self.super_enc.inverse_transform(y_pred)

            if super_category == 'JCB':
                x_train_JCB.append(x)
                y_train_JCB.append(y)
            elif super_category == 'RH':
                x_train_RH.append(x)
                y_train_RH.append(y)
            elif super_category == 'RC':
                x_train_RC.append(x)
                y_train_RC.append(y)
            continue
        x_train_JCB, y_train_JCB = np.array(x_train_JCB), np.array(y_train_JCB)
        x_train_RH, y_train_RH = np.array(x_train_RH), np.array(y_train_RH)
        x_train_RC, y_train_RC = np.array(x_train_RC), np.array(y_train_RC)
        print(x_train_JCB.shape, y_train_JCB.shape)

        if len(y_train_JCB) >0:
            self.sub_classifierJCB.build(input_shape=x_train_JCB[0].shape, classes=10)
            self.sub_classifierJCB_hist = self.sub_classifierJCB.fit(x_train_JCB, y_train_JCB,\
                    batch_size=self.batch_size,\
                    epochs=epochs
                        )
        if len(y_train_RH) >0:
            self.sub_classifierRH.build(input_shape=x_train_RH[0].shape, classes=10)
            self.sub_classifierRH_hist = self.sub_classifierRH.fit(x_train_RH, y_train_RH,\
                    batch_size=self.batch_size,\
                    epochs=epochs
                        )
        if len(x_train_RC) > 0:
            self.sub_classifierRC.build(input_shape=x_train_RC[0].shape, classes=10)
            self.sub_classifierRC_hist = self.sub_classifierRC.fit(x_train_RC, y_train_RC,\
                    batch_size=self.batch_size,\
                    epochs=epochs
                        )

    def save_model(self, obj, filename):
        with open(filename+'.pickle', 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    def load_model(self, filename):
        model = None
        with open(filename+'.pickle', 'rb') as output:  # Overwrites any existing file.
            model = pickle.load(output)
        return model

# %%
