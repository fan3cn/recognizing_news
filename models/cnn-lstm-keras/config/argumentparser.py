from __future__ import print_function

import argparse

def ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='data source')
    parser.add_argument('--embedding_file_path', type=str, default='cc.zh.300.vec',
                        help='path to file for embedding vectors')
    parser.add_argument('--model_dir', type=str, default='model',
                        help='directory to store checkpointed models')
    parser.add_argument('--nb_words', type=int, default=10000,
                        help='Number of words to keep from the dataset')
    parser.add_argument('--max_sequence_len', type=int, default=300,
                        help='Maximum input sequence length')
    parser.add_argument('--validation_split', type=float, default=0.4,
                        help='Fraction of data to be used for validation')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of the embedding space to be used')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Dimension of Fully-Connect Layer')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout prob')
    parser.add_argument('--model_name', type=str, default='cnn-simple',
                        help='Name of the model variant, two options: CNN-simple, LSTM')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.0,
                        help='decay rate for rmsprop')
    parser.add_argument('--run_all', type=int, default=0,
                        help='Try all combinations')
    parser.add_argument('--baseline', type=str, default='',
                        help='Compare with the baseline.')
    parser.add_argument('--use_word_embedding', type=int, default=0,
                        help='Use word embedding or not')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='Use early stopping or not')
    return parser.parse_args()
