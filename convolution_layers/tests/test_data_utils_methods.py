import unittest

import numpy as np

from data_utils import clean_x_test
from data_utils import clean_y_test
from data_utils import get_confusion_matrix
from data_utils import clean_prediction


class TestDataUtilsMethods(unittest.TestCase):
    def test_clean_x_test(self):
        size_max = 10
        binary_words = np.array([
            np.reshape(np.array(list(bytes('Hallo', 'utf-8').zfill(size_max))), (size_max, 1)),
            np.reshape(np.array(list(bytes('Mensch', 'utf-8').zfill(size_max))), (size_max, 1))
        ])
        print(binary_words)
        print(clean_x_test(binary_words))
        assert clean_x_test(binary_words) == ['Hallo', 'Mensch']

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
            ['True label\\prediction label',       'male', 'neutral', 'female'],
            ['                        male',    0,      0,         1],
            ['                     neutral', 0,      0,         0],
            ['                      female',  0,      0,         1]]

        assert matrix == expected_matrix

    def test_clean_prediction(self):
        prediction = np.array([
            [0.01, 0.97, 0.02],
            [0.1, 0.3, 0.6]
        ])

        assert clean_prediction(prediction) == ['neutral', 'female']

if __name__ == '__main__':
    unittest.main()
