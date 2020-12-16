# %%
import csv
import glob
import os
import shutil
from tqdm import tqdm
from sklearn.utils import shuffle
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pydot
import scipy
import tensorflow as tf
from IPython.display import SVG
from keras import layers
from keras.initializers import glorot_uniform
from keras.layers import (Activation, Add, AveragePooling2D,
                          BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          GlobalMaxPooling2D, Input, MaxPooling2D,
                          ZeroPadding2D)
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import layer_utils, plot_model
from keras.utils.vis_utils import model_to_dot
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from pydub import AudioSegment
from scipy import misc
import pandas as pd


class AudioGenerator():

    def __init__(self):
        self.data_dir = 'processed_data'
        self.genres = 'pop blues country disco classical hiphop jazz metal reggae rock'.split()

    def create_dirs(self):
        os.makedirs('%s/train/audio_features' % self.data_dir, exist_ok=True)
        os.makedirs('%s/test/audio_features' % self.data_dir, exist_ok=True)

    def generate_audio_csv(self, ):
        header = 'filename_wav filename_png genre chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
        for i in range(1, 21):
            header += f' mfcc{i}'
        header += ' label'
        header = header.split()

        for mode in ['train', 'test']:
            file_path = '%s/%s/audio_features/audio_features.csv' % (
                self.data_dir, mode)

            file = open(file_path, 'w', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(header)

            for g in self.genres:
                for filename_wav in tqdm(os.listdir(self.data_dir+'/'+mode+'/split_audio/'+g), desc=g):
                    songname = '%s/%s/split_audio/%s/%s' % (
                        self.data_dir, mode, g, filename_wav)
                    y, sr = librosa.load(songname, mono=True, duration=30)
                    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                    rms = librosa.feature.rms(y=y)
                    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                    zcr = librosa.feature.zero_crossing_rate(y)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr)

                    filename_png = filename_wav[:-3] + 'png'
                    filename_png = f'{self.data_dir}/{mode}/spectrograms/{g}/{filename_png}'
                    to_append = f'{filename_wav} {filename_png} {g} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
                    for e in mfcc:
                        to_append += f' {np.mean(e)}'
                    to_append += f' {g}'
                    file = open('%s/%s/audio_features/audio_features.csv' %
                                (self.data_dir, mode), 'a', newline='')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())
            df = pd.read_csv(file_path)
            df = shuffle(df)
            df.to_csv(file_path)

    def split_test_files(self):
        train_dir = f'{self.data_dir}/train/split_audio/'
        print('Splitting test files...')
        for g in self.genres:
            filenames = os.listdir(os.path.join(train_dir, f"{g}"))
            test_files = filenames[0:100]

            for f in test_files:
                shutil.move(
                    train_dir + f"{g}" + "/" + f, f'{self.data_dir}/test/split_audio/{g}/{f}')
        print('Splitting test files done!')


# %%
