# coding: utf-8

from collections import Counter
import numpy as np
import tensorflow.contrib.keras as kr

def build_vocab(train_dir, vocab_dir, vocab_size=10000):
  data_train, _ = read_file(train_dir)
  all_data = []
  for content in data_train:
    all_data.extend(content)
  counter = Counter(all_data)
  count_pairs = counter.most_common(vocab_size - 1)
  words, _ = list(zip(*count_pairs))
  words = ['<PAD>'] + list(words)
  with open(vocab_dir, mode='w') as fout:
    for word in words:
      fout.write(word+'\n')

def read_vocab(vocab_dir):
  with open(vocab_dir,'r') as fp:
    words = [_.strip() for _ in fp.readlines()]
  word_to_id = dict(zip(words, range(len(words))))
  return words, word_to_id

def read_category():
  categories = [0,1,2]
  cat_to_id = dict(zip(categories, range(len(categories))))
  return categories, cat_to_id

def read_file_test(filename):
  contents=[]
  with open(filename,'r') as dataset:
    for line in dataset:
      words=line.split(' ')
      contents.append(words[0:-1])
  return contents

def process_file_test(filename, word_to_id, max_length=400):
  contents = read_file_test(filename)
  data_id = []
  for i in range(len(contents)):
    if(len(contents[i])<=max_length): data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
    else:
      l=int(max_length/2)
      r=len(contents[i])-l
      while(r<len(contents[i])-max_length/4):
        if(contents[i][r]=='。' or contents[i][r]=='!' or contents[i][r]=='？'):break
        r=r+1
      while(l>max_length/4):
        if(contents[i][l]=='。' or contents[i][l]=='!' or contents[i][l]=='？'):break
        l=l-1
      if(r>len(contents[i])-max_length/4): r=len(contents[i])-int(max_length/2)
      if(l<max_length/4): l=int(max_length/2)-1
      data_id.append([word_to_id[x] for x in contents[i][0:l+1] if x in word_to_id])
      data_id[-1].extend([word_to_id[x] for x in contents[i][r:] if x in word_to_id])
  # padding
  x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
  return x_pad

def read_file(filename):
  labels=[]
  contents=[]
  with open(filename,'r') as dataset:
    for line in dataset:
      items=line.split('\t')
      labels.append(int(items[0]))
      words=items[1].split(' ')
      contents.append(words[0:-1])
  return contents,labels

def process_file(filename, word_to_id, cat_to_id, max_length=400):
  contents, labels = read_file(filename)
  data_id, label_id = [], []
  for i in range(len(contents)):
    if(len(contents[i])<=max_length): data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
    else:
      l=int(max_length/2)
      r=len(contents[i])-l
      while(r<len(contents[i])-max_length/4):
        if(contents[i][r]=='。' or contents[i][r]=='!' or contents[i][r]=='？'):break
        r=r+1
      while(l>max_length/4):
        if(contents[i][l]=='。' or contents[i][l]=='!' or contents[i][l]=='？'):break
        l=l-1
      if(r>len(contents[i])-max_length/4): r=len(contents[i])-int(max_length/2)
      if(l<max_length/4): l=int(max_length/2)-1
      data_id.append([word_to_id[x] for x in contents[i][0:l+1] if x in word_to_id])
      data_id[-1].extend([word_to_id[x] for x in contents[i][r:] if x in word_to_id])
    label_id.append(cat_to_id[labels[i]])
  # padding
  x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
  y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # one-hot representation
  return x_pad, y_pad

def batch_iter(x, y, batch_size=64):
  """batch generation"""
  data_len = len(x)
  num_batch = int((data_len - 1) / batch_size) + 1
  indices = np.random.permutation(np.arange(data_len))
  indices=indices%data_len
  x_shuffle = x[indices]
  y_shuffle = y[indices]
  for i in range(num_batch):
    start_id = i * batch_size
    end_id = min((i + 1) * batch_size, data_len)
    yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
