from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D
from keras.optimizers import SGD, RMSprop

from matplotlib import pyplot

from extract_data import get_data_sets
from extract_data import byte_to_string

PADDING = 20

TRAINING_PERCENTAGE = 0.8

(x_train, y_train), (x_test, y_test) = get_data_sets(PADDING, TRAINING_PERCENTAGE)

model = Sequential()

model.add(Dense(80, activation='sigmoid', input_dim=PADDING))
model.add(Dropout(0.1))
model.add(Dense(80, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['acc', 'mean_squared_error'])

history = model.fit(x_train, y_train,
          epochs=20,
          batch_size=150)

score = model.evaluate(x_test, y_test, batch_size=150)

prediction = model.predict(x_test[:100])

[print(pair) for pair in zip(prediction, byte_to_string(x_test[:100]), y_test)]

print(score)

pyplot.plot(history.history['acc'])
pyplot.show()
