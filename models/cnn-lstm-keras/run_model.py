# If u want to obtain reproducible results please import reproduciable;
# But it will significantly slow down the training speed.
#import reproduciable
import numpy as np
import load_data as ld
import keras
from model.model import model_selector
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, Flatten, GlobalMaxPooling1D, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import argparse
import gensim
import time
import os
from config import argumentparser

def main():
    #args = argparse.ArgumentParser().parse_args()
    args = argumentparser.ArgumentParser()
    if args.run_all == 1:
        # Try all combinations
        #models = ['cnn-rand', 'cnn-static', 'cnn-non-static', 'cnn-3']
        models = ['cnn-3', 'cnn-non-static', 'cnn-static','cnn-rand']
        #data_dirs = ['data/', '/home/fanyy/sohu/raw_data/','/home/fanyy/sohu/raw_data_short/', '/home/fanyy/sohu/raw_data_short/stopwords/']
        data_dirs = ['data/', '/home/fanyy/sohu/raw_data/','/home/fanyy/sohu/raw_data_short/']
        nb_words = [10000]
        max_sequence_len = [200, 500]
        batch_size = [64]

        for model in models:
            for data_dir in data_dirs:
                for nb in nb_words:
                    for max_len in max_sequence_len:
                        for b_size in batch_size:
                            args.model_name = model
                            args.data_dir = data_dir
                            args.nb_words = nb
                            args.max_sequence_len = max_len
                            args.batch_size = b_size
                            try:
                                run(args)
                            except:
                                print("exception!!!")
                            else:
                                print("OK!")
    else:
        run(args)

def run(args):
    # Data path
    path = args.data_dir
    #path = "data/"
    X_train = os.path.join(path, 'X.train')
    Y_train = os.path.join(path, 'Y.train')
    X_online_test = os.path.join(path, 'X.test')
    all_text_path = os.path.join(path, 'all_text')
    id_test = os.path.join(path, 'id.test')

    # Seed
    seed = 13
    # fix random seed for reproducibility
    np.random.seed(seed)
    #print("Reading all text...")
    all_texts = open( all_text_path ).readlines()
    #print("Tokenizing...")
    tokenizer = Tokenizer(num_words=args.nb_words)
    #print("Fitting...")
    tokenizer.fit_on_texts(all_texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    print("Loading training data...")
    X_train, y_train = ld.load_data(shulf=True, X_train = X_train, Y_train = Y_train, tokenizer=tokenizer, max_len=args.max_sequence_len)

    # Select Model
    model = model_selector(args, word_index)
    print(model.summary())
    # Callback list
    callbacks = []
    #filepath = "weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
    #check_point = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    if args.early_stop == 1:
        eraly_stop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=0, verbose=0, mode='auto')
        callbacks.append(eraly_stop)

    print("Training...")
    r = model.fit(X_train, y_train, epochs = args.num_epochs, batch_size = args.batch_size, callbacks=callbacks, validation_split = args.validation_split)
    # Find out the best validation accuracy
    best_val_acc = max(r.history['val_acc'])
    if(best_val_acc < 0.69):
        print("Low Val_Acc.")
    # load X.validate
    X_validate, y_validate = ld.load_data(X_train=X_online_test, Y_train=None, tokenizer=tokenizer, max_len=args.max_sequence_len)
    # predict
    y_validate = model.predict(X_validate, verbose=0)
    # Convert from Categorical to numberical
    y_validate = np.argmax(y_validate, axis=1)
    # Compare with the baseline
    count_diff = 0;
    count_same = 0;
    if args.baseline:
        with open(args.baseline) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                if y_validate[i] == int(lines[i].split('\t')[1]):
                    count_same += 1
                else:
                    count_diff += 1
    print("Same:%d" % count_same)
    # Load validation Ids
    ids = np.loadtxt(id_test, dtype=bytes).astype(str)
    assert(len(ids) == len(y_validate))
    # Generate result
    result = [ [ids[i] + "\t"+ str(y_validate[i]) + "\t" + "NULL" + "\t" + "NULL"] for i in range(len(y_validate))]
    # Time
    ts = time.strftime("%Y%m%d-%H%M%S",time.localtime())
    # Format output
    output = ts + ("_acc_%.2f_%s_%s_%d_%d_%d_%d_%d_%d" % 
        (best_val_acc * 100, args.model_name, path.replace("/","-"), 
            args.nb_words, args.max_sequence_len, args.embedding_dim, args.batch_size, count_same, args.use_word_embedding) )
    np.savetxt( "result_" + output + ".txt", result, fmt='%s')
    # Done
    print("Done!")

if __name__ == "__main__":
    main()
