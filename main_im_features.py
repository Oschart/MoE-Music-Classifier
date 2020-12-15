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
# %%
os.makedirs('image_data/spectrograms3sec')
os.makedirs('image_data/spectrograms3sec/train')
os.makedirs('image_data/spectrograms3sec/test')
#%%
wav_dir = 'GTZAN/genres_original/'
genres = 'hiphop jazz metal reggae rock'.split()
# %%
for g in genres:
    path_audio ='image_data/audio3sec/'+g
    os.makedirs(path_audio)
    path_train = 'image_data/spectrograms3sec/train/'+g
    path_test = 'image_data/spectrograms3sec/test/'+g
    os.makedirs(path_train)
    os.makedirs(path_test)
# %%
from pydub import AudioSegment
i = 0
for g in genres:
    j=0
    print(f"{g}")
    for filename in os.listdir(wav_dir+'/'+g):
        songname = wav_dir+g+'/'+filename
        j = j+1
        for w in range(0,10):
            i = i+1
            #print(i)
            t1 = 3*(w)*1000
            t2 = 3*(w+1)*1000
            newAudio = AudioSegment.from_wav(songname)
            new = newAudio[t1:t2]
            new.export(f'image_data/audio3sec/{g}/{g+str(j)+str(w)}.wav', format="wav")
# %%
for g in genres:
    j = 0
    print(g)
    for filename in os.listdir('image_data/audio3sec/'+g):
        song  =  'image_data/audio3sec/'+g+'/'+filename
        j = j+1
        y,sr = librosa.load(song,duration=3)
        mels = librosa.feature.melspectrogram(y=y,sr=sr)
        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        print(j)
        p = plt.imshow(librosa.power_to_db(mels,ref=np.max))
        plt.savefig(f'image_data/spectrograms3sec/train/{g}/{g+str(j)}.png')
# %%
directory = "image_data/spectrograms3sec/train/"
for g in genres:
    filenames = os.listdir(os.path.join(directory,f"{g}"))
    random.shuffle(filenames)
    test_files = filenames[0:100]

    for f in test_files:
        shutil.move(directory + f"{g}"+ "/" + f,"/image_data/spectrograms3sec/test/" + f"{g}")

# %%
train_dir = "/image_data/spectrograms3sec/train/"
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(288,432),color_mode="rgba",class_mode='categorical',batch_size=128)

validation_dir = "/image_data/spectrograms3sec/test/"
vali_datagen = ImageDataGenerator(rescale=1./255)
vali_generator = vali_datagen.flow_from_directory(validation_dir,target_size=(288,432),color_mode='rgba',class_mode='categorical',batch_size=128)

# %%
def GenreModel(input_shape = (288,432,4),classes=9):
    X_input = Input(input_shape)
    X = Conv2D(8,kernel_size=(3,3),strides=(1,1))(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(16,kernel_size=(3,3),strides = (1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(32,kernel_size=(3,3),strides = (1,1)))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(64,kernel_size=(3,3),strides=(1,1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)

    X = Conv2D(128,kernel_size=(3,3),strides=(1,1))(X)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2))(X)


    X = Flatten()(X)

    X = Dropout(rate=0.3)

    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    model = Model(inputs=X_input,outputs=X,name='GenreModel')

    return model