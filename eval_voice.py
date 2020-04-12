from keras.models import load_model
from misc import load_images
import json
import sys
import numpy as np
import os
import soundfile as sf
from scipy.signal import spectrogram
from misc import process_specs, split_specs, make_mel_filter_bank


if len(sys.argv) < 4:
    print("usage: python eval_voice.py KERAS_MODEL INPUT_FOLDER OUTPUT_FILE")
    sys.exit(1)


folder = sys.argv[2]
if folder[0] != '/':
    folder += '/'
file_names = sorted(os.listdir(folder))
file_names = list(filter(lambda x: '.wav' in x, file_names))
file_names = list(map(lambda x: x.split('.')[0], file_names))

# load normalization
with open('voice_normalization.json', 'r') as f:
    norm = json.load(f)

# open output file
output = open(sys.argv[3], 'w')

# get model
model = load_model(sys.argv[1])

# prepare mel filter
M = make_mel_filter_bank(120, 513, 8000)

# for all voice files, evaluate them
for file_name in file_names:
    x, fs = sf.read(folder + file_name + ".wav")
    f, t, spec = spectrogram(x, fs=fs, window='blackman', nfft=1024, noverlap=150, nperseg=500)
    data = process_specs([spec], M)
    data = np.array(split_specs(data))
    data -= norm['mean']
    data /= norm['std']
    predicted = model.predict(data)
    predicted = np.mean(predicted)
    output.write(file_name + " " + str(predicted) + " " + str(int(predicted > 0.35)) + "\n")

output.close()