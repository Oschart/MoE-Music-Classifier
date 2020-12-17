# %%
from operator import concat
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
from utils import get_superclass_mapping
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
comb = Combiner()
comb.build_experts()
# %%
spect_hist = comb.train_spectro_expert(epochs=1)
x_train, x_test, y_train, y_test, _, _ = comb.concat_spect_aud()
terminal_hist, loss, acc = comb.train_terminal_expert(x_train, x_test, y_train, y_test, epochs=1000)
# %%
# print(hist.history.keys())
x_train, x_test, y_train, y_test= comb.get_audio_ft()
audio_hist, loss, acc = comb.train_audio_expert(x_train, x_test,\
        y_train, y_test, epochs=50)
# accuracy 71 on audio only
# %%
# superclassifier training
mapping = get_superclass_mapping()
super_hist, loss, acc = comb.train_superclassifier(mapping=mapping, concat=False)

