import unittest

import numpy as np

from extract_data import clean_x_test
from extract_data import clean_y_test
from extract_data import get_confusion_matrix
from extract_data import clean_prediction


class TestStringMethods(unittest.TestCase):
    def test_clean_x_test(self):
        binary_words = np.array([
            bytes('__Hallo', 'utf-8'),
            bytes('_Mensch', 'utf-8')
        ])

        assert (clean_x_test(binary_words) == ['Hallo', 'Mensch'])

    def test_clean_y_test(self):
        triples = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ])

        actual = clean_y_test(triples)

        assert (type(actual) is list)

        assert (actual == ['female', 'neutral'])

    def test_get_confusion_matrix(self):
        actual_y = np.array(['female', 'male'])
        predicted_y = np.array(['female', 'female'])

        matrix = get_confusion_matrix(actual_y, predicted_y)

        expected_matrix = [
            ['',       'male', 'neutral', 'female'],
            ['male',    0,      0,         1],
            ['neutral', 0,      0,         0],
            ['female',  0,      0,         1]]

        assert matrix == expected_matrix

    def test_clean_prediction(self):
        prediction = np.array([
            [0.01, 0.97, 0.02],
            [0.1, 0.3, 0.6]
        ])

        assert clean_prediction(prediction) == ['neutral', 'feminine']

if __name__ == '__main__':
    unittest.main()
