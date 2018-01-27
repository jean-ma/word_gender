from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM
from keras.optimizers import RMSprop

import sys

from letter_embedding.data_utils import get_data_sets
from letter_embedding.data_utils import generate_report
from letter_embedding.data_utils import generate_fit_evolution_figure
from letter_embedding.data_utils import new_report_directory
from letter_embedding.data_utils import interactive_test


if len(sys.argv) == 2:
    model_relative_path = sys.argv[1]

    model = load_model(model_relative_path)

    interactive_test(model)

    sys.exit(0)

TRAINING_PERCENTAGE = 0.8

(x_train, y_train), (x_test, y_test), alphabet_size = get_data_sets(TRAINING_PERCENTAGE)

model = Sequential()

model.add(Embedding(alphabet_size + 1, output_dim=20, name='embedding'))
model.add(LSTM(20, dropout=0))
model.add(Dense(3, activation='sigmoid'))

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['acc', 'mean_squared_error'])

fit_history = model.fit(x_train, y_train,
                        epochs=20,
                        batch_size=15)

test_score = model.evaluate(x_test, y_test, batch_size=15)

prediction = model.predict(x_test)

print('Report creation...')

dir_name = new_report_directory()

model.save(dir_name + 'model.h5')

generate_report(test_score, x_test, y_test, prediction, dir_name)

generate_fit_evolution_figure(fit_history, dir_name)

print('Report in ' + dir_name)
