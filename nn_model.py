from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D
from keras.optimizers import SGD, RMSprop

from matplotlib import pyplot

from extract_data import get_data_sets
from extract_data import generate_report
from extract_data import generate_fit_evolution_figure
from extract_data import new_report_directory

TRAINING_PERCENTAGE = 0.8

(x_train, y_train), (x_test, y_test), padding = get_data_sets(TRAINING_PERCENTAGE)

model = Sequential()

model.add(Dense(150, activation='sigmoid', input_dim=padding))
model.add(Dropout(0.1))
model.add(Dense(150, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['acc', 'mean_squared_error'])

fit_history = model.fit(x_train, y_train,
                        epochs=500,
                        batch_size=150)

test_score = model.evaluate(x_test, y_test, batch_size=150)

prediction = model.predict(x_test)

print('Report creation...')

dir_name = new_report_directory()

generate_report(test_score, x_test, y_test, prediction, dir_name)

generate_fit_evolution_figure(fit_history, dir_name)

print('Report in ' + dir_name)
