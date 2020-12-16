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

from SpectroGenerator import SpectroGenerator
from SpectroExpert import SpectroExpert

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# %%
print("trying git")
spec_gen = SpectroGenerator()
spec_gen.create_dirs()
# %%
spec_gen.generate_wav_segs(n_split=10)

# %%
spec_gen.generate_spectrograms()
# %%
spec_gen.split_test_files()

# %%
train_dir = "image_data/spectrograms3sec/train/"
train_datagen = ImageDataGenerator(rescale=1./255)
img_size = (250, 400)
img_size_rgba = (*img_size, 4)
batch_size = 64
train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, color_mode="rgba", class_mode='categorical', batch_size=batch_size)

validation_dir = "image_data/spectrograms3sec/test/"
vali_datagen = ImageDataGenerator(rescale=1./255)
vali_generator = vali_datagen.flow_from_directory(
    validation_dir, target_size=img_size, color_mode='rgba', class_mode='categorical', batch_size=batch_size)


# %%
print(train_generator.class_indices)
spec_exp = SpectroExpert()
spec_exp.build(input_shape=img_size_rgba, learning_rate=0.005, classes=10)
spec_exp.fit_gen(train_generator, epochs=70,
                 validation_data=vali_generator, shuffle=True)
# %%


# %%
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

# %%
