from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
import soundfile as sf
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw
from mel_bank_filter import make_mel_filter_bank
import os
import json


def filter_mask(sp, right_side=10, count=4):
    energy = np.sum(sp[12:,:], axis=0)
    mask = (energy >= (np.max(energy) - np.min(energy))*0.25 +  np.min(energy)).astype('int')
    new_mask = np.zeros((len(mask))).astype('int')

    for i in range(0, len(mask)-right_side):
        if ((mask[i] == 1 or new_mask[i-1] == 1) and np.sum(mask[i:i+right_side]) >= count):
            for j in reversed(range(i,i+right_side)):
                if mask[j] == 1:
                    new_mask[i:j] = 1
                    break
    return new_mask


def window_indeces(mask, window_step=12, margin=5):
    indeces = np.array([])
    for i in range(1, len(mask)):
        if mask[i] == 1 and mask[i-1] == 0:
            for j in range(i, len(mask)):
                if mask[j] == 0 or j == len(mask)-1:
                    if j-i <= window_step:
                        indeces = np.append(indeces, [int((i+j)/2)])
                    else:
                        indeces = np.append(indeces, np.arange(i+margin, j, window_step))
                    break
    return indeces.astype('int')


def get_windows(sp, indeces, window_left=12, window_right=12):
    windows = []
    for i in indeces:
        if i >= window_left and i + window_right < sp.shape[1]:
            w = sp[:,i-window_left:i+window_right]
            w.shape = (w.shape[0], w.shape[1], 1)
            windows.append(w)
    return windows


def load_specs(folders):
    spec_list = []
    if type(folders) is str:
        folders = [folders]
    for folder in folders:
        if folder[0] != '/':
            folder = folder + '/'
        wavs = os.listdir(folder)
        for wav in sorted(wavs):
            if '.wav' in wav:
                x, fs = sf.read(folder + wav)
                f, t, sp = spectrogram(x, fs=fs, window='blackman', nfft=1024, noverlap=150, nperseg=500)
                spec_list.append(sp)
    return spec_list


def process_specs(spec_list, M):
    new_spec_list = []
    for s in spec_list:
        n = s[:,50:-50]
        n = n*n
        n = np.matmul(n.T, M.T).T
        n = np.log10(n)
        new_spec_list.append(n)
    return new_spec_list


def split_specs(all_spec):
    new_all_spec = []
    for spec in all_spec:
        """
        plt.subplot(2,1,1)
        plt.pcolormesh(spec)
        """
        mask = filter_mask(spec)
        idxs = window_indeces(mask)
        windows = get_windows(spec, idxs)
        new_all_spec.extend(windows)
        """
        for i, w in enumerate(windows):
            plt.subplot(2,len(windows),i+1+len(windows))
            plt.axis('off')
            plt.pcolormesh(w)
        plt.show()
        """
    return new_all_spec


M = make_mel_filter_bank(120, 513, 8000)

# load and prepare non target data
non_target_data = load_specs(['non_target_dev', 'non_target_train'])
non_target_data = process_specs(non_target_data, M)
non_target_data = split_specs(non_target_data)
non_target_val_data = np.array(non_target_data[5300:])
non_target_data = np.array(non_target_data[:5300])
# make labels
non_target_labels = np.zeros((len(non_target_data)))
non_target_val_labels = np.zeros((len(non_target_val_data)))

# load and prepare target data
target_data = load_specs(['target_dev', 'target_train'])
target_data = process_specs(target_data, M)
target_data = split_specs(target_data)
target_val_data = np.array(target_data[390:])
target_data = np.array(target_data[:390])
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
model.fit(data, labels, batch_size=64, epochs=20, validation_data=(val_data, val_labels), shuffle=True)