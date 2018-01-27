from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalAveragePooling1D
from keras.optimizers import RMSprop

from convolution_layers.data_utils import get_data_sets
from convolution_layers.data_utils import generate_report
from convolution_layers.data_utils import generate_fit_evolution_figure
from convolution_layers.data_utils import new_report_directory

TRAINING_PERCENTAGE = 0.8

(x_train, y_train), (x_test, y_test), padding = get_data_sets(TRAINING_PERCENTAGE)

model = Sequential()

model.add(Conv1D(10, 2, strides=1, padding='valid', input_shape=(padding, 1), name='conv 1'))
model.add(Conv1D(10, 2, strides=1, padding='valid', name='conv 2'))
model.add(Conv1D(10, 2, strides=1, padding='valid', name='conv 3'))
model.add(GlobalAveragePooling1D())
model.add(Dense(20, activation='sigmoid', name='dense 1'))
model.add(Dense(3, activation='softmax', name='dense 2'))

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['acc', 'mean_squared_error'])

fit_history = model.fit(x_train, y_train,
                        epochs=500,
                        batch_size=16)

test_score = model.evaluate(x_test, y_test, batch_size=16)

prediction = model.predict(x_test)

print('Report creation...')

dir_name = new_report_directory()

model.save(dir_name + 'model.h5')

generate_report(test_score, x_test, y_test, prediction, dir_name)

generate_fit_evolution_figure(fit_history, dir_name)

print('Report in ' + dir_name)
