from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import json
import warnings


def load_data(X_train=None, Y_train=None, shulf = None, tokenizer=None, max_len=500):

    x_train = tokenizer.texts_to_sequences(open(X_train).readlines())
    x_train = pad_sequences(x_train, maxlen=max_len)
    print(x_train.shape)

    #np.random.seed()
    if Y_train:
        labels_train = np.loadtxt(Y_train)
    else:
        labels_train = np.zeros(len(x_train))
    print(labels_train.shape)

    if shulf:
        # shuffle
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train = x_train[indices]
        labels_train = labels_train[indices]
    # Convert to categorical
    labels_train = to_categorical(labels_train, num_classes=None)
    return x_train, labels_train


