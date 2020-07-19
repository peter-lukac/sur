"""
Face training for CNN
author: Peter Lukac
login: xlukac11
April 2020
"""

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
import numpy as np
import json
import sys
from misc import load_images


if len(sys.argv) != 3 and len(sys.argv) != 4:
    sys.stderr.write("usage: python train_face.py TARGET_FOLDER NON_TARGET_FOLDER [KERAS_MODEL]\n")
    sys.exit(1)


target_folder = sys.argv[1]
non_target_folder = sys.argv[2]


non_target_data = load_images(non_target_folder, True)
# split data into train data and validation data
non_target_val_data = non_target_data[164:]
non_target_data = non_target_data[:164]
# make labels
non_target_labels = np.zeros((len(non_target_data)))
non_target_val_labels = np.zeros((len(non_target_val_data)))


target_data = load_images(target_folder, True)
# split data into train data and validation data
target_val_data = target_data[25:]
target_data = target_data[:25]
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

with open('face_normalization.json', 'w') as f:
    json.dump({'mean': float(data_mean), 'std':float(data_std)}, f)


# create CNN
model = Sequential()

model.add(Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=(80,80,1)))
model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train
model.fit(data, labels, batch_size=1, epochs=20, validation_data=(val_data, val_labels), shuffle=True)

if len(sys.argv) == 4:
    model.save(sys.argv[3])

