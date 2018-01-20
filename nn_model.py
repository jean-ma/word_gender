import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D
from keras.optimizers import SGD

from matplotlib import pyplot

from numpy.random import shuffle
from keras import metrics

from extract_data import get_words
from extract_data import byte_to_string
from extract_data import get_sets

import numpy as np

PADDING = 15

TRAINING = 8  # 8 out of 10 go to the training set, the rest to the test set

words = get_words()
shuffle(words)

(xy_train, xy_test) = get_sets(words, TRAINING)
(x_train, y_train) = zip(*xy_train)
x_train = np.array(x_train)
y_train = keras.utils.to_categorical(y_train, 3)
#y_train = keras.utils.to_categorical(np.random.randint(3, size=(len(x_train), 1)), num_classes=3)

(x_test, y_test) = zip(*xy_test)
x_test = np.array(x_test)
y_test = keras.utils.to_categorical(y_test, 3)

model = Sequential()

model.add(Dense(64, activation='relu', input_dim=PADDING))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['acc', 'mean_squared_error'])

history = model.fit(x_train, y_train,
          epochs=20,
          batch_size=10)

score = model.evaluate(x_test, y_test, batch_size=10)

prediction = model.predict(x_test[:100])

[print(pair) for pair in zip(prediction, byte_to_string(x_test[:100]))]

print(score)

pyplot.plot(history.history['acc'])
pyplot.show()
