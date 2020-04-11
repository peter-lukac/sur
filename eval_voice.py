from keras.models import load_model
from misc import load_images
import json
import sys
import numpy as np
import os
import soundfile as sf
from scipy.signal import spectrogram


if len(sys.argv) < 4:
    print("usage: python eval_voice.py KERAS_MODEL INPUT_FOLDER OUTPUT_FILE")
    sys.exit(1)


folder = sys.argv[2]
if folder[0] != '/':
    folder += '/'
file_names = sorted(os.listdir(folder))
file_names = list(filter(lambda x: '.wav' in x, file_names))
file_names = list(map(lambda x: x.split('.')[0], file_names))


# for all voice files, evaluate them
for file_name in file_names:
    x, fs = sf.read(folder + file_name + ".wav")
    f, t, spec = spectrogram(x, fs=fs, window='blackman', nfft=1024, noverlap=150, nperseg=500)