"""
Voice training for CNN
author: Peter Lukac
login: xlukac11
April 2020
"""

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
import numpy as np
import json
from misc import (filter_mask, window_indeces, get_windows,
                load_specs, process_specs, split_specs, make_mel_filter_bank)
import sys

from numpy.random import seed
seed(42)

from tensorflow.compat.v2.random import set_seed
set_seed(42)


if len(sys.argv) != 3 and len(sys.argv) != 4:
    sys.stderr.write("usage: python train_voice.py TARGET_FOLDER NON_TARGET_FOLDER [KERAS_MODEL]\n")
    sys.exit(1)

target_folder = sys.argv[1]
non_target_foler = sys.argv[2]

M = make_mel_filter_bank(120, 513, 8000)

# load and prepare non target data
non_target_data = load_specs(non_target_foler)
non_target_data = process_specs(non_target_data, M)
non_target_data = split_specs(non_target_data)
non_target_val_data = np.array(non_target_data[5300:])
non_target_data = np.array(non_target_data[:5300])
# make labels
non_target_labels = np.zeros((len(non_target_data)))
non_target_val_labels = np.zeros((len(non_target_val_data)))

# load and prepare target data
target_data = load_specs(target_folder)
target_data = process_specs(target_data, M)
target_data = split_specs(target_data)
target_val_data = np.array(target_data[360:])
target_data = np.array(target_data[:360])
# make labels
target_labels = np.ones((len(target_data)))
target_val_labels = np.ones((len(target_val_data)))

# concatenate data
data = np.concatenate((target_data, non_target_data))
val_data = np.concatenate((target_val_data, non_target_val_data))

# concatenate labels
labels = np.concatenate((target_labels, non_target_labels))
val_labels = np.concatenate((target_val_labels, non_target_val_labels))

# normalize data
all_data = np.concatenate((data, val_data))
data_mean = np.mean(all_data)
data -= data_mean
val_data -= data_mean
data_std = np.std(all_data)
data /= data_std
val_data /= data_std


with open('voice_normalization.json', 'w') as f:
    json.dump({'mean': float(data_mean), 'std':float(data_std)}, f)


# create CNN
model = Sequential()

model.add(Conv2D(16, kernel_size=(5,5), activation='relu', input_shape=(120,24,1)))
model.add(Conv2D(16, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(4,2)))

model.add(Conv2D(32, kernel_size=(5,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,1)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train
model.fit(data.astype('float32'), labels, batch_size=64, epochs=12, validation_data=(val_data, val_labels), shuffle=True)


if len(sys.argv) == 4:
    model.save(sys.argv[3])