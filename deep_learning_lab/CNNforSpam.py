import numpy as np  # linear algebra
import pandas as pd  # data processing
from subprocess import check_output

# Any results we write to the current directory are saved as output.
DATA_FILE = 'spam.csv'
df = pd.read_csv(DATA_FILE, encoding='latin-1')
print(df.head())

tags = df.v1
texts = df.v2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
import time
from keras import metrics

print('import done')
num_max = 1000
# preprocessing
le = LabelEncoder()
tags = le.fit_transform(tags)
tok = Tokenizer(num_words=num_max)
tok.fit_on_texts(texts)
mat_texts = tok.texts_to_matrix(texts, mode='count')
print(tags[:5])
print(mat_texts[:5])
print(tags.shape, mat_texts.shape)


# trying a simple model first

def get_simple_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(num_max,)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc', metrics.binary_accuracy])
    print('compile done')
    return model


def check_model(model, x, y):
    model.fit(x, y, batch_size=32, epochs=10, verbose=1, validation_split=0.2)


m = get_simple_model()
check_model(m, mat_texts, tags)
# for cnn preprocessing
max_len = 100
cnn_texts_seq = tok.texts_to_sequences(texts)
print(cnn_texts_seq[0])
cnn_texts_mat = sequence.pad_sequences(cnn_texts_seq, maxlen=max_len)
print(cnn_texts_mat[0])
print(cnn_texts_mat.shape)


def get_cnn_modelv1():
    model = Sequential()
    # start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    # 1000 is num_max
    model.add(Embedding(1000,
                        20,
                        input_length=max_len))
    model.add(Dropout(0.1))
    model.add(Conv1D(64,
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('relu'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc', metrics.binary_accuracy])
    return model


m = get_cnn_modelv1()
check_model(m, cnn_texts_mat, tags)


def get_cnn_modelv2():  # added embed
    model = Sequential()
    model.add(Embedding(1000,
                        50,
                        input_length=max_len))
    model.add(Dropout(0.1))
    model.add(Conv1D(64,
                     3,
                     padding='valid',
                     activation='tanh',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('relu'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc', metrics.binary_accuracy])
    return model


m = get_cnn_modelv2()
check_model(m, cnn_texts_mat, tags)


def get_cnn_modelv3():  # added filter
    model = Sequential()

    model.add(Embedding(1000,
                        20,
                        input_length=max_len))
    model.add(Dropout(0.1))
    model.add(Conv1D(256,
                     3,
                     padding='valid',
                     activation='tanh',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc', metrics.binary_accuracy])
    return model


m = get_cnn_modelv3()
check_model(m, cnn_texts_mat, tags)