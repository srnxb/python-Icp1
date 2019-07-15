!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip

from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Embedding
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import TensorBoard


LOG_DIR = './log'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)

get_ipython().system_raw('./ngrok http 6006 &')

! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

df = pd.read_csv('imdb_master.csv',encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values


#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
#getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)

# max_review_len= max([len(s.split()) for s in sentences])
# vocab_size= len(tokenizer.word_index)+1
# sentences = tokenizer.texts_to_sequences(sentences)
# padded_docs= pad_sequences(sentences,maxlen=max_review_len)
#model.add(Embedding(2000, 50, input_length=2000))


le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

# Number of features
#print(input_dim)
model = Sequential()
# model.add(layers.Dense(300,input_dim=500, activation='relu'))
model.add(layers.Dense(100,activation='sigmoid'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
tbCallBack = TensorBoard(log_dir='./log', histogram_freq=1,
                          write_graph=True,
                          write_grads=True,
                          batch_size=256,
                         write_images=True)

model.fit(X_train, y_train,
          batch_size=256,
          epochs=2,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[tbCallBack])

# history=model.fit(X_train,y_train, epochs=3, verbose=True, validation_data=(X_test,y_test), batch_size=256)

[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# summarize history for accuracy
N = 3
#plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
