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
import shutil
from keras.preprocessing.image import ImageDataGenerator
import random



class SpectroGenerator():

    def __init__(self):
        self.wav_dir = 'GTZAN/genres_original/'
        self.genres = 'pop blues country disco classical hiphop jazz metal reggae rock'.split()

    def create_dirs(self):
        os.makedirs('image_data/spectrograms3sec')
        os.makedirs('image_data/spectrograms3sec/train')
        os.makedirs('image_data/spectrograms3sec/test')

        for g in self.genres:
            path_audio ='image_data/audio3sec/'+g
            os.makedirs(path_audio)
            path_train = 'image_data/spectrograms3sec/train/'+g
            path_test = 'image_data/spectrograms3sec/test/'+g
            os.makedirs(path_train)
            os.makedirs(path_test)

    def generate_wav_segs(self):
        from pydub import AudioSegment
        i = 0
        for g in self.genres:
            j=0
            print(f"{g}")
            for filename in os.listdir(self.wav_dir+'/'+g):
                songname = self.wav_dir+g+'/'+filename
                j = j+1
                for w in range(0,3):
                    i = i+1
                    #print(i)
                    t1 = 3*(w)*1000
                    t2 = 3*(w+1)*1000
                    newAudio = AudioSegment.from_wav(songname)
                    new = newAudio[t1:t2]
                    new.export(f'image_data/audio3sec/{g}/{g+str(j)+str(w)}.wav', format="wav")

    def generate_spectrograms(self):
        for g in self.genres:
            j = 0
            print(g)
            for filename in os.listdir('image_data/audio3sec/'+g):
                song  =  'image_data/audio3sec/'+g+'/'+filename
                j = j+1
                y,sr = librosa.load(song,duration=3)
                mels = librosa.feature.melspectrogram(y=y,sr=sr)
                fig = plt.Figure()
                #canvas = FigureCanvas(fig)
                print(j)
                #p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
                plt.savefig(f'image_data/spectrograms3sec/train/{g}/{g+str(j)}.png')

    def split_test_files(self):
        train_dir = "image_data/spectrograms3sec/train/"
        for g in self.genres:
            filenames = os.listdir(os.path.join(train_dir,f"{g}"))
            random.shuffle(filenames)
            test_files = filenames[0:100]

            for f in test_files:
                shutil.move(train_dir + f"{g}"+ "/" + f,"image_data/spectrograms3sec/test/" + f"{g}")

                
# %%
