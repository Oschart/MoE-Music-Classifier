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
from Combiner import Combiner

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
comb = Combiner()
comb.build_experts()

# %%
comb.train_experts(epochs=1)


# %%
print(comb.train_dfx.columns)

# %%
batch_index = 0
data_list = []
while batch_index <= comb.spec_train_gen.batch_index:
    x, y = comb.spec_train_gen.next()
    x, y = comb.audio_ft_train.next()
    data_list.append(x)
    batch_index = batch_index + 1

# %%
