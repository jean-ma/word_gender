import re
import numpy as np
from matplotlib import pyplot
from numpy.random import shuffle
from datetime import datetime
from sklearn.metrics import confusion_matrix
import os

import keras


ROOT_PATH = './data/'
FILENAME = 'nomen.sql'
NEUTRAL = 'neutral.txt'
FEM = 'female.txt'
MASC = 'male.txt'

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
                bytes_word = bytes(splitted[0].strip('\''), 'utf-8')

                size_max = max([size_max, len(bytes_word)])

                if splitted[1] == '\'SUB:NOM:SIN:MAS\'':
                    categorized_words.append((bytes_word, CONST_MAS))

                if splitted[1] == '\'SUB:NOM:SIN:FEM\'':
                    categorized_words.append((bytes_word, CONST_FEM))

                if splitted[1] == '\'SUB:NOM:SIN:NEU\'':
                    categorized_words.append((bytes_word, CONST_NEU))

    return categorized_words, size_max


def split_sets(words=np.array([]), training=0.8):
    size_training = int(len(words) * training)
    return words[:size_training], words[size_training:]


def get_data_sets(training_percentage=0.8):
    (word_gender, size_max) = get_words()
    word_gender = np.array([
        np.array([np.array(list(w.zfill(size_max))), gender])
        for w, gender in word_gender])
    shuffle(word_gender)

    (xy_train, xy_test) = split_sets(word_gender, training_percentage)
    xy_train = boost_minorities(xy_train)
    (x_train, y_train) = zip(*xy_train)
    x_train = np.array(x_train)
    y_train = keras.utils.to_categorical(y_train, 3)

    (x_test, y_test) = zip(*xy_test)
    x_test = np.array(x_test)
    y_test = keras.utils.to_categorical(y_test, 3)

    return (x_train, y_train), (x_test, y_test), size_max


def boost_minorities(xy_train=np.array([])):
    male = []
    female = []
    neutral = []

    for (x_train, y_train) in xy_train:
        if y_train == CONST_MAS:
            male.append((x_train, y_train))
        if y_train == CONST_NEU:
            neutral.append((x_train, y_train))
        if y_train == CONST_FEM:
            female.append((x_train, y_train))

    max_size = max([len(male), len(female), len(neutral)])

    for gender in [male, female, neutral]:
        while len(gender) < max_size:
            diff_size = max_size - len(gender)
            gender += gender[:diff_size]

    completed = np.array(male + female + neutral)
    np.random.shuffle(completed)

    return completed


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
            yield ''.join(bytes(w.tolist()).decode('utf-8'))

    return [w.strip('0') for w in byte_to_string(x_test)]


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

    with open(report_filename, 'w') as report:
        accuracy = 1
        report.write('Overall accuracy: {}\n'.format(test_score[accuracy]))
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
