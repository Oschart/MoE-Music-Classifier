# %%
import glob
import os
import random
import shutil

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


class SpectroGenerator():

    def __init__(self):
        self.wav_dir = 'GTZAN/genres_original/'
        self.genres = 'pop blues country disco classical hiphop jazz metal reggae rock'.split()
        self.data_dir = 'processed_data'

    def create_dirs(self):
        os.makedirs('%s/train/spectrograms' % self.data_dir, exist_ok=True)
        os.makedirs('%s/test/spectrograms' % self.data_dir, exist_ok=True)

        for mode in ['train', 'test']:
            for g in self.genres:
                path_audio = '%s/%s/split_audio/%s' % (self.data_dir, mode, g)
                os.makedirs(path_audio, exist_ok=True)
                path = '%s/%s/spectrograms/%s' % (self.data_dir, mode, g)
                os.makedirs(path, exist_ok=True)

    def generate_wav_segs(self, n_split=3):
        for g in self.genres:
            i = 0
            print(f"{g}")
            for filename in os.listdir(self.wav_dir+'/'+g):
                songname = self.wav_dir+g+'/'+filename
                for w in range(0, n_split):
                    i = i+1
                    # print(i)
                    t1 = (30//n_split)*(w)*1000
                    t2 = (30//n_split)*(w+1)*1000
                    newAudio = AudioSegment.from_wav(songname)
                    new = newAudio[t1:t2]
                    new.export(
                        f'{self.data_dir}/train/split_audio/{g}/{g+str(i)}.wav', format="wav")

    def generate_spectrograms(self):
        for g in self.genres:
            j = 0
            print(g)
            for filename in os.listdir(f'{self.data_dir}/train/split_audio/{g}'):
                song = f'{self.data_dir}/train/split_audio/{g}/{filename}'
                j = j+1
                y, sr = librosa.load(song, duration=3)
                mels = librosa.feature.melspectrogram(y=y, sr=sr)
                fig = plt.Figure()
                canvas = FigureCanvas(fig)
                print(j)
                p = plt.imshow(librosa.power_to_db(mels, ref=np.max))
                plt.savefig(
                    f'{self.data_dir}/train/spectrograms/{g}/{g+str(j)}.png')
                plt.cla()

    def split_test_files(self):
        train_dir = f'{self.data_dir}/train/spectrograms/'
        for g in self.genres:
            filenames = os.listdir(os.path.join(train_dir, f"{g}"))
            random.shuffle(filenames)
            test_files = filenames[0:100]

            for f in test_files:
                shutil.move(
                    train_dir + f"{g}" + "/" + f, f'{self.data_dir}/test/spectrograms/{g}')


# %%
