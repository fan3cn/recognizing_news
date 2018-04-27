### Introduction  
Based on millions of real-world data provided by Sohu, the goal of the competition is to recognize low-quality news, such as marketing, vulgarity and the click bait. The dataset contains both labeled and unlabeled records composed of news, text fragments and pictures which account in a volume of130GB in total.
We first preprocess the raw data(html pages crawled from the web) by applying data cleaning, Chinese word segmentation and word embedding. Next we conducted experiments on various models, one typical method is based on bag-of-words, such as Fasttext and some traditional machine learning algorithms(SVM, shallow Neural Network, Logistic Regression, Naïve Bayes) based on hand-crafted features. Another state-of-art technique representative in NLP task is focusing on semantic understanding, such as CNN(convolutional neural network), LSTM are widely used in text classification for the purpose of automatic feature engineering.

### Prerequisites
- Linux/Macos(tested on Redhat)
- Python3.5/2.7
- Numpy
- Tensorflow
- Keras
- awk/sed...
- fasttext
- THULAC

### How to run?

- Data preprocessing
```shell
sh preprocess.sh
```

- Install THULAC  
Please go to [thulac.org](http://thulac.thunlp.org/) for instruction

- Word segmentation
```shell
python word_seg.py
```

- Run models  
`cd models/`  
Choose a model that you are interested in, check out the instruction in `readme.md` before you run that model.
