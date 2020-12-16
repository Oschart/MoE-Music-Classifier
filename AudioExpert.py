import librosa
import numpy as np


class AudioExpert():
    def __init__(self):
        self.file_path = None
    def fit(self):
        return None
    def predict(self, file_path='image_data/audio3sec/blues/blues10.wav'):
        y, sr = librosa.load(file_path, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)     
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        features_ls = [np.mean(chroma_stft), np.mean(rms),\
                np.mean(spec_cent), np.mean(spec_bw),\
                np.mean(rolloff), np.mean(zcr)]    
        for e in mfcc:
            features_ls.append(np.mean(e))
        return features_ls

