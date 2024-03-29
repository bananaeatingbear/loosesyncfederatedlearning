
# coding: utf-8

import numpy as np
import pandas as pd
import pickle
from scipy import misc
from tqdm import tqdm

def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

meta = unpickle('cifar-100-python/meta')

fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]

train = unpickle('cifar-100-python/train')

filenames = [t.decode('utf8') for t in train[b'filenames']]
fine_labels = train[b'fine_labels']
data = train[b'data']

images = list()
for d in data:
    image = np.zeros((32,32,3), dtype=np.uint8)
    image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
    image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
    image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
    images.append(image)

images=np.array(images)
print (images.shape)
np.save('cifar100_train_data.npy',images)
fine_labels = np.array(fine_labels)
print (fine_labels.shape)
np.save('cifar100_train_label.npy',fine_labels)

test = unpickle('cifar-100-python/test')

filenames = [t.decode('utf8') for t in test[b'filenames']]
fine_labels = test[b'fine_labels']
data = test[b'data']

images = list()
for d in data:
    image = np.zeros((32,32,3), dtype=np.uint8)
    image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
    image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
    image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
    images.append(image)

images=np.array(images)
print (images.shape)
np.save('cifar100_test_data.npy',images)
fine_labels = np.array(fine_labels)
print (fine_labels.shape)
np.save('cifar100_test_label.npy',fine_labels)