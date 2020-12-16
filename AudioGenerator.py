# %%
import tensorflow as tf
import numpy as np
import scipy
from scipy import misc
import glob
from PIL import Image
import os
import matplotlib.pyplot as plt
import librosa
from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten,
                          Conv2D, AveragePooling2D, Dropout, MaxPooling2D, GlobalMaxPooling2D)
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment
from keras.preprocessing.image import ImageDataGenerator
import csv


class AudioGenerator():

    def __init__(self):
        self.data_dir = 'processed_data'
        self.genres = 'pop blues country disco classical hiphop jazz metal reggae rock'.split()

    def create_dirs(self):
        os.makedirs('%s/train/audio_features' % self.data_dir, exist_ok=True)
        os.makedirs('%s/test/audio_features' % self.data_dir, exist_ok=True)

    def generate_audio_csv(self, ):
        header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
        for i in range(1, 21):
            header += f' mfcc{i}'
        header += ' label'
        header = header.split()

        for mode in ['train', 'test']:
            file = open('%s/%s/audio_features/audio_features.csv' %
                        (self.data_dir, mode), 'w', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(header)
            
            for g in self.genres:
                print(g)
                for filename in os.listdir(self.data_dir+mode+'/split_audio/'+g):
                    songname = '%s/%s/split_audio/%s/%s' % (
                        self.data_dir, mode, g, filename)
                    y, sr = librosa.load(songname, mono=True, duration=30)
                    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                    rms = librosa.feature.rms(y=y)
                    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                    zcr = librosa.feature.zero_crossing_rate(y)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr)
                    to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
                    for e in mfcc:
                        to_append += f' {np.mean(e)}'
                    to_append += f' {g}'
                    file = open('%s/%s/audio_features/audio_features.csv' %
                                (self.data_dir, mode), 'a', newline='')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())
