from __future__ import print_function

from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.layers import Dense, Input, Flatten, Dropout, Merge, Activation
from keras.models import Model, Sequential
from keras.optimizers import Adadelta, RMSprop
import gensim
import numpy as np
import io

# Global dict 
wv_model = {}

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        #Python 2&3 compatibility 
        #data[tokens[0]] = map(float, tokens[1:])
        data[tokens[0]] = tokens[1:]
    return data



def model_selector(args, word_index):
    '''Method to select the model to be used for classification'''
    print('Defining model.')
    model = Sequential()
    embedding_layer = get_embedding_layer(args, args.use_word_embedding, word_index)     
    model.add(Embedding(args.nb_words, args.embedding_dim, input_length=args.max_sequence_len))
    if (args.model_name.lower() == 'cnn-simple'):
        model.add(Dropout(args.dropout))
        model.add(Conv1D(256, 5, padding='valid', activation='relu', strides=1))
        model.add(MaxPooling1D(5))
        model.add(Flatten())
        model.add(Dense(args.hidden_dim, activation='relu'))
        model.add(Dropout(args.dropout))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif (args.model_name.lower() == 'lstm'):
        model.add(LSTM(100))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model




def get_embedding_layer(args, use_embedding,  word_index):
    '''Get the embedding layer according to whether using word embedding or not'''
    if (use_embedding == 1):
        print("Using Word Embedding...")
        print("Loading embeddings...")
        #wv_model = gensim.models.Word2Vec.load('word_dim_60.model')
        global wv_model
        if not wv_model:
            wv_model = load_vectors(args.embedding_file_path)
        print("Constructing embedding matrix...")
        for k,v in wv_model.items():
            args.embedding_dim = len(v)
            break

        embedding_matrix = np.zeros((args.nb_words + 1, args.embedding_dim))

        for word, i in word_index.items():
            if i > args.nb_words:
                continue
            if  word in wv_model and wv_model[word] is not None:
                embedding_matrix[i] = wv_model[word]

        embedding_layer = Embedding(args.nb_words + 1,
                                    args.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=args.max_sequence_len,
                                    trainable=False)
    else:
        embedding_layer = Embedding(args.nb_words + 1,
                                    args.embedding_dim,
                                    weights=None,
                                    input_length=args.max_sequence_len,
                                    trainable=False)
    return embedding_layer

