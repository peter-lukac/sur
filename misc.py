from scipy.signal import spectrogram
import soundfile as sf
import numpy as np
import os
import cv2


def load_images(folders, grey_cale=True):
    img_list = []
    if type(folders) is str:
        folders = [folders]
    for folder in folders:
        if folder[0] != '/':
            folder = folder + '/'
        imgs = os.listdir(folder)
        for img in sorted(imgs):
            if '.png' in img:
                #print("loading: " + folder + img)
                if grey_cale:
                    i = cv2.imread(folder + img, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    i.shape = (80,80,1)
                    img_list.append(i)
                else:
                    img_list.append(cv2.imread(folder + img).astype(np.float32))
    return img_list


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


def hz_to_mel(hz):
    return 1127*np.log(1+hz/700)

def mel_to_hz(mel):
    return 700*np.exp(mel/1127)-700

def make_mel_filter_bank(number_of_banks, fft_size, max_f, normalize=True):
    bank_filter = np.zeros((number_of_banks, fft_size))
    frequencies = np.linspace(0, hz_to_mel(max_f), number_of_banks+2)
    frequencies = mel_to_hz(frequencies)
    f_indexes = np.round((frequencies/max_f) * (fft_size-1)).astype('int')

    for i in range(0, number_of_banks):
        bank_filter[i, f_indexes[i]:f_indexes[i+1]+1] = np.linspace(0,1,f_indexes[i+1] - f_indexes[i]+2)[1:]
        bank_filter[i, f_indexes[i+1]+1:f_indexes[i+2]+1] = np.linspace(1,0,f_indexes[i+2] - f_indexes[i+1]+2)[1:-1]

        if normalize:
            bank_filter[i] = bank_filter[i] / np.sum(bank_filter[i])

    return bank_filter