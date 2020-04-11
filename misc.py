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