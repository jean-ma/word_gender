import re
import numpy as np


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

def get_words():
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
                word = np.array([letter for letter in padding(bytes(splitted[0].strip('\''), 'utf-8'), 15)])

                if splitted[1] == '\'SUB:NOM:SIN:MAS\'':
                    categorized_words.append(np.array([word, CONST_MAS]))

                if splitted[1] == '\'SUB:NOM:SIN:FEM\'':
                    categorized_words.append(np.array([word, CONST_FEM]))

                if splitted[1] == '\'SUB:NOM:SIN:NEU\'':
                    categorized_words.append(np.array([word, CONST_NEU]))

    return np.array(categorized_words)


def byte_to_string(words=np.array([])):
    return [''.join(bytes(w.tolist()).decode('utf-8')) for w in words]


def padding(word=bytes(), padding=30):
    """

    :param word: word
    :param padding: padding to apply
    :return: a list of words with given padding
    """
    return word[:padding].rjust(padding, bytes('_', 'ascii'))


def get_sets(words=np.array([]), training=10):
    size_training = int(len(words) / 10 * training)
    return words[:size_training], words[size_training:]
