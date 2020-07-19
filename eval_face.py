"""
Face evaluation using CNN
author: Peter Lukac
login: xlukac11
April 2020
"""

from keras.models import load_model
from misc import load_images
import json
import sys
import numpy as np
import os


if len(sys.argv) != 4:
    sys.stderr.write("usage: python eval_face.py KERAS_MODEL INPUT_FOLDER OUTPUT_FILE\n")
    sys.exit(1)


# load data and file names
data = np.array(load_images(sys.argv[2], True))
file_names = sorted(os.listdir(sys.argv[2]))
file_names = list(filter(lambda x: '.png' in x, file_names))
file_names = list(map(lambda x: x.split('.')[0], file_names))

# load normalization values
with open('face_normalization.json', 'r') as f:
    norm = json.load(f)

# normalize data
data -= norm['mean']
data /= norm['std']

# load model
model = load_model(sys.argv[1])

# predict
values = model.predict(data)

# save results
with open(sys.argv[3], 'w') as f:
    for file_name, value in zip(file_names, values):
        f.write(file_name + " " + str(value[0]) + " " + str(int(value>0.5)) + "\n")
