import re
import numpy as np
from keras.models import Sequential
from matplotlib import pyplot
from numpy.random import shuffle
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from keras.preprocessing.sequence import pad_sequences
import os

import keras


ROOT_PATH = './data/'
FILENAME = 'nomen.sql'
NEUTRAL = 'neutral.txt'
FEM = 'female.txt'
MASC = 'male.txt'

german_alphabet = 'abcdefghijklmnopqrstuvwxyzäöüß'
letter_to_number = dict([(letter, idx) for letter, idx in zip(german_alphabet, range(1, len(german_alphabet)+1))])

REPORT_FILENAME = 'report.txt'

FIT_HISTORY_FILENAME = 'fit_history.png'

CONST_MAS = 0
CONST_NEU = 1
CONST_FEM = 2

TRAINING = ROOT_PATH + 'training/'
VALIDATION = ROOT_PATH + 'validation/'


def get_words():
    """

    :return: feminine, masculine, neutral
    """
    with open(ROOT_PATH + FILENAME, 'r') as all_words:

        read_data = all_words.read()

        m = re.finditer('\(.*?\)', read_data)

        categorized_words = []
        size_max = 0

        for match in m:
            splitted = match.group().split(',')[2:4]

            if len(splitted) > 1:
                german_word = splitted[0].strip('\'').lower()

                exotic_word = len([True for l in german_word if l not in german_alphabet]) > 0

                if not exotic_word:
                    size_max = max([size_max, len(german_word)])

                    if splitted[1] == '\'SUB:NOM:SIN:MAS\'':
                        categorized_words.append((german_word, CONST_MAS))

                    if splitted[1] == '\'SUB:NOM:SIN:FEM\'':
                        categorized_words.append((german_word, CONST_FEM))

                    if splitted[1] == '\'SUB:NOM:SIN:NEU\'':
                        categorized_words.append((german_word, CONST_NEU))

    return categorized_words, size_max


def split_sets(words=np.array([]), training=0.8):
    size_training = int(len(words) * training)
    return words[:size_training], words[size_training:]


def get_data_sets(training_percentage=0.8):
    (word_gender, size_max) = get_words()
    word_gender = np.array([
        np.array([format_row(w), gender])
        for w, gender in word_gender])
    shuffle(word_gender)

    (xy_train, xy_test) = split_sets(word_gender, training_percentage)
    (x_train, y_train) = zip(*xy_train)
    x_train = pad_sequences(np.array(x_train), maxlen=size_max)
    y_train = keras.utils.to_categorical(y_train, 3)

    (x_test, y_test) = zip(*xy_test)
    x_test = pad_sequences(np.array(x_test), maxlen=size_max)
    y_test = keras.utils.to_categorical(y_test, 3)

    return (x_train, y_train), (x_test, y_test), size_max


def clean_prediction(predictions=np.array([])):
    def prediction_to_category(row=np.array([])):
        if row[0] > row[1] and row[0] > row[2]:
            return 'male'
        elif row[1] > row[0] and row[1] > row[2]:
            return 'neutral'
        else:
            return 'female'

    return [
        prediction_to_category(row)
        for row in predictions
    ]


def clean_x_test(x_test=np.array([])):
    def byte_to_string(words):
        for w in words:
            yield ''.join([german_alphabet[number-1] for number in w.tolist() if number > 0])

    return [w for w in byte_to_string(x_test)]


def clean_y_test(y_test=np.array([])):
    triplet_to_string = {
        CONST_MAS: 'male',
        CONST_FEM: 'female',
        CONST_NEU: 'neutral'
    }

    def gender(row=np.array([])):
        return triplet_to_string[row[0]*0 + row[1]*1 + row[2]*2]

    return [gender(row) for row in y_test]


def generate_report(test_score, x_test, y_test, prediction, directory_name):
    cleaned_prediction = clean_prediction(prediction)
    cleaned_x_test = clean_x_test(x_test)

    cleaned_y_test = clean_y_test(y_test)

    report_filename = directory_name + REPORT_FILENAME

    confusion_mat = get_confusion_matrix(np.array(cleaned_y_test), np.array(cleaned_prediction))

    f1 = f1_score(np.array(cleaned_y_test), np.array(cleaned_prediction), average='micro')

    with open(report_filename, 'w') as report:
        accuracy = 1
        report.write('Overall accuracy: {}\n'.format(test_score[accuracy]))
        report.write('F1 score: {}\n'.format(f1))
        report.write('Confusion matrix: \n')

        [report.write('{}, {}, {}, {}\n'.format(*row)) for row in confusion_mat]

        [
            report.write(pred + ';{};{}\n'.format(x, y))
            for pred, x, y in zip(cleaned_prediction, cleaned_x_test, cleaned_y_test)
        ]


def get_confusion_matrix(actual=np.array([]), predicted=np.array([])):
    conf = confusion_matrix(actual, predicted, labels=['male', 'neutral', 'female']).tolist()
    return [['True label\\prediction label', 'male', 'neutral', 'female'],
            ['                        male'] + conf[0],
            ['                     neutral'] + conf[1],
            ['                      female'] + conf[2]]


def generate_fit_evolution_figure(fit_history, directory_name):
    filename = directory_name + FIT_HISTORY_FILENAME
    pyplot.title('Prediction Accuracy over epochs')
    pyplot.xlabel('Epochs')
    pyplot.ylabel('Accuracy')
    pyplot.plot(fit_history.history['acc'])
    pyplot.savefig(filename)


def new_report_directory():
    now = datetime.today().strftime('%Y-%m-%dT%H%M')
    dir_name = ROOT_PATH + 'report_' + now + '/'
    os.mkdir(dir_name)
    return dir_name


def format_row(row):
    return np.array([letter_to_number[letter] for letter in row])


def interactive_test(model=Sequential()):
    print('type "q" to quit')
    _, max_length = model.input_shape

    testing_word = input()
    while testing_word != "q":
        formatted_x = pad_sequences(np.array([format_row(testing_word)]), maxlen=max_length)

        prediction = model.predict(formatted_x)

        first_column = clean_prediction(np.array([[1, 0, 0]]))
        second_column = clean_prediction(np.array([[0, 1, 0]]))
        third_column = clean_prediction(np.array([[0, 0, 1]]))

        print('{} {} {}'.format(first_column, second_column, third_column))
        print('{} ({})'.format(str(prediction[0]), str(clean_prediction(prediction)[0])))
        testing_word = input()
