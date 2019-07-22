import numpy as np # linear algebra
import pandas as pd # processing data
import matplotlib.pyplot as plt

# we need to fit model with sequence of tokens with specific length
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
# normal LSTM/GRU
from keras.layers import Dense, Embedding, GRU, LSTM, Dropout, Bidirectional
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, rmsprop

# keras wrapper for k-fold cross-validation
from keras.wrappers.scikit_learn import KerasClassifier
# normsl cross validation
from sklearn.model_selection import cross_val_score, train_test_split
# cross validation for tuning hyperparameter
from sklearn.model_selection import GridSearchCV

import os

x_raw = []
y_raw = []

with open("spam.csv", encoding = "ISO-8859-1") as f:
    for line in f:
        y_raw.append(line.split()[0])
        x_raw.append(' '.join(i for i in line.split()[1:]))
y = [1 if i=='ham' else 0 for i in y_raw]

print(max(len(s) for s in x_raw))
print(min(len(s)for s in x_raw))
sorted_X = sorted(len(s) for s in x_raw)
print(sorted_X[len(sorted_X) // 2])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_raw)
sequences = tokenizer.texts_to_sequences(x_raw)

vocab_size = len(tokenizer.word_index)+1
print(vocab_size)
# divide sum of length of all sequences by number of all sequences to find averge length of each sequence
sum([len(x) for x in sequences]) // len(sequences)
pad = 'post'
max_len = 25
embedding_size = 100
batch_size = 32
sequences = pad_sequences(sequences, maxlen=max_len, padding=pad, truncating=pad)
sequences.shape

X_train, X_test, y_train, y_test = train_test_split(sequences, y, test_size = 0.3, random_state= 0)
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
model.add(Dropout(0.1))
model.add(LSTM(140, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(1, activation='tanh', name='Classification'))
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.1, callbacks=[callback_early_stopping])
n_epochs = 10
results = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
# model evaluation
eval_ = model.evaluate(X_test, y_test)
print(eval_[0], eval_[1]) # loss / accuracy