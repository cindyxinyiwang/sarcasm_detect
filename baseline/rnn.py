# recurrent neural networks
from keras.models import Sequential
from keras.layers import Input, LSTM, Embedding, Dense, merge
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.engine.topology import Merge
from keras.layers.core import Reshape
import numpy as np


pos_data, neg_data, data = [], [], []
pos_count, neg_count = 0, 0
max_len = 20

with open("pos_data/pos0.txt") as myfile:
	for line in myfile:
		data.append(one_hot(line, 10000, lower=True))
		pos_count += 1
with open("neg_data/neg0.txt") as myfile:
	for line in myfile:
		data.append(one_hot(line, 10000, lower=True))
		neg_count += 1

data = pad_sequences(data, maxlen=max_len, dtype='float32')
data = np.array(data)

dataX = np.reshape(data, (len(data), max_len, 1))

labels = [1 for i in range(pos_count)] + [0 for i in range(neg_count)]
labels = np.array(labels)


left = Sequential()
left.add(LSTM(32, input_shape=(dataX.shape[1], dataX.shape[2]), init='uniform',
	inner_init='uniform', forget_bias_init='one', return_sequences=False, activation='tanh',
	inner_activation='sigmoid'))
#left.add(Reshape((len(data), 32 * max_len)))

right = Sequential()
right.add(LSTM(32, input_shape=(dataX.shape[1], dataX.shape[2]), init='uniform',
	inner_init='uniform', forget_bias_init='one', return_sequences=False, activation='tanh',
	inner_activation='sigmoid', go_backwards=True))
#right.add(Reshape((len(data), 32 * max_len)))

model = Sequential()
model.add(Merge([left, right], mode='ave'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit([dataX, dataX], labels, nb_epoch=10, batch_size=1, verbose=2)

		