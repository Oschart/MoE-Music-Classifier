import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
from PIL import Image
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
import keras

import warnings
class AudioExpert():
    def __init__(self):
        self.x_train = None
        self.y_train = None
        
    def fit(self, songs_dir='GTZAN/genres_original/',\
         genres = 'blues classical country disco hiphop jazz metal pop reggae rock',\
         output_dir='GTZAN/audio_features.csv'):
        self.songs_dir = songs_dir
        self.genres = genres
        self.output_dir = output_dir
    def predict(self, file_path=None):
        if file_path is None:
            header = 'filename chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
            for i in range(1, 21):
                header += f' mfcc{i}'
            header += ' label'
            header = header.split()

            file = open(self.output_dir, 'w', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(header)
            genres = self.genres.split()
            for g in genres:
                for filename in os.listdir(self.songs_dir + g):
                    songname = self.songs_dir + g + '/' + filename
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
                    file = open(self.output_dir, 'a', newline='')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())
        else:
            y, sr = librosa.load(file_path, mono=True, duration=30)
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)     
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            features_ls = [np.mean(chroma_stft), np.mean(rms), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]    
            for e in mfcc:
                features_ls.append(np.mean(e))
            return features_ls

