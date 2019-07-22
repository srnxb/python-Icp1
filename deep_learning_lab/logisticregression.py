from tensorboardcolab import *
from __future__ import print_function
import os
from datetime import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import metrics
# from keras.regularizers import l1l2
# from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model
from sklearn.model_selection import train_test_split

tbc = TensorBoardColab()

df = pd.read_csv('heart.csv')
kc_data = pd.DataFrame(df,
                       columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
                                "slope", "ca", "thal", "target"])
label_col = 'ca'
print(kc_data.describe())

kc_X_train, kc_X_valid, kc_Y_train, kc_Y_valid = train_test_split(kc_data.iloc[:, 0:13], kc_data.iloc[:, 13],
                                                                  test_size=0.3, random_state=87)
np.random.seed(155)


def norm_stats(df1, df2):
    dfs = df1.append(df2)
    minimum = np.min(dfs)
    maximum = np.max(dfs)
    mu = np.mean(dfs)
    sigma = np.std(dfs)
    return (minimum, maximum, mu, sigma)


def z_score(col, stats):
    m, M, mu, s = stats
    df2 = pd.DataFrame()
    for c in col.columns:
        df2[c] = (col[c] - mu[c]) / s[c]
    return df2


stats = norm_stats(kc_X_train, kc_X_valid)
arr_X_train = np.array(z_score(kc_X_train, stats))
arr_Y_train = np.array(kc_Y_train)
arr_X_valid = np.array(z_score(kc_X_valid, stats))
arr_Y_valid = np.array(kc_Y_valid)
print('Training shape:', arr_X_train.shape)
print('ddd', arr_Y_train.shape)
print('Training samples: ', arr_X_train.shape[0])
print('Validation samples: ', arr_X_valid.shape[0])


def model1(X_size, Y_size):
    model = Sequential()
    model.add(Dense(100, activation='tanh', input_shape=(X_size,)))
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    # model.add(Dense(100, activation="tanh", input_shape=(X_size,)))
    model.add(Dense(50, activation="tanh"))
    model.add(Dense(Y_size))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=[metrics.mae])
    return (model)


# model2 is different from basic_model_1 but doing the same task with different structure
def model2(X_size, Y_size):
    # reg = l1l2(l1=0.01, l2=0.01)
    model = Sequential()
    model.add(Dense(100, activation='tanh', input_shape=(X_size,)))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(Y_size))
    keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    optimizers = ['rmsprop', 'adam']
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=[metrics.mae])
    return (model)


model = model2(arr_X_train.shape[1], 1)

model.summary()
epochs = 15
batch_size = 32

history = model.fit(arr_X_train, arr_Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True,
                    verbose=2,  # Change it to 2, if wished to observe execution
                    validation_data=(arr_X_valid, arr_Y_valid), callbacks=[TensorBoardColabCallback(tbc)])
train_score = model.evaluate(arr_X_train, arr_Y_train, verbose=0)
valid_score = model.evaluate(arr_X_valid, arr_Y_valid, verbose=0)

print('Train MAE: ', round(train_score[1], 4), ', Train Loss: ', round(train_score[0], 4))
print('Val MAE: ', round(valid_score[1], 4), ', Val Loss: ', round(valid_score[0], 4))

keras_callbacks = [
    ModelCheckpoint('/tmp/keras_checkpoints/model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                    save_best_only=True, verbose=2),
    ModelCheckpoint('/tmp/keras_checkpoints/model.{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True,
                    verbose=0),
    TensorBoard(log_dir='./model_3', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0,
                embeddings_layer_names=None, embeddings_metadata=None),
    EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=0)
]


def plot_histogram(h, Xsize=6, Ysize=10):
    # Preparing plot
    fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [Xsize, Ysize]
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)

    # summarizing history for MAE
    plt.subplot(211)
    plt.plot(h['mean_absolute_error'])
    plt.plot(h['val_mean_absolute_error'])
    plt.title('Training vs Validation MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # summarizing history for loss
    plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot it in IPython (non-interactive)
    plt.draw()
    plt.show()

    return


plot_histogram(history.history, Xsize=8, Ysize=12)
score = model.evaluate(arr_X_valid, arr_Y_valid)
print('test accuracy', score[1])