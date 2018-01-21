import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D
from keras.optimizers import SGD

from matplotlib import pyplot

from extract_data import get_data_sets
from extract_data import byte_to_string

PADDING = 20

TRAINING_PERCENTAGE = 0.8

(x_train, y_train), (x_test, y_test) = get_data_sets(PADDING, TRAINING_PERCENTAGE)

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
