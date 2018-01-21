import re
import numpy as np
from numpy.random import shuffle
import keras


ROOT_PATH = './data/'
FILENAME = 'nomen.sql'
NEUTRAL = 'neutral.txt'
FEM = 'feminine.txt'
MASC = 'masculine.txt'

CONST_MAS = 0
CONST_NEU = 1
CONST_FEM = 2

TRAINING = ROOT_PATH + 'training/'
VALIDATION = ROOT_PATH + 'validation/'


def get_words(padding=30):
    """

    :return: feminine, masculine, neutral
    """
    with open(ROOT_PATH + FILENAME, 'r') as all_words:

        read_data = all_words.read()

        m = re.finditer('\(.*?\)', read_data)

        categorized_words = []

        for match in m:
            splitted = match.group().split(',')[2:4]

            if len(splitted) > 1:
                word = np.array([
                    letter
                    for letter in padd(bytes(splitted[0].strip('\'')[-(padding - 1) :], 'utf-8'), padding)
                ])

                if splitted[1] == '\'SUB:NOM:SIN:MAS\'':
                    categorized_words.append(np.array([word, CONST_MAS]))

                if splitted[1] == '\'SUB:NOM:SIN:FEM\'':
                    categorized_words.append(np.array([word, CONST_FEM]))

                if splitted[1] == '\'SUB:NOM:SIN:NEU\'':
                    categorized_words.append(np.array([word, CONST_NEU]))

    return np.array(categorized_words)


def byte_to_string(words=np.array([])):
    return [''.join(bytes(w.tolist()).decode('utf-8')) for w in words]


def padd(word=bytes(), padding=30):
    """

    :param word: word
    :param padding: padding to apply
    :return: a list of words with given padding
    """
    return word[-padding:].rjust(padding, bytes('_', 'ascii'))


def split_sets(words=np.array([]), training=0.8):
    size_training = int(len(words) * training)
    return words[:size_training], words[size_training:]


def get_data_sets(padding=20, training_percentage=0.8):
    words = get_words(padding)
    shuffle(words)

    (xy_train, xy_test) = split_sets(words, training_percentage)
    (x_train, y_train) = zip(*xy_train)
    x_train = np.array(x_train)
    y_train = keras.utils.to_categorical(y_train, 3)

    (x_test, y_test) = zip(*xy_test)
    x_test = np.array(x_test)
    y_test = keras.utils.to_categorical(y_test, 3)

    return (x_train, y_train), (x_test, y_test)
