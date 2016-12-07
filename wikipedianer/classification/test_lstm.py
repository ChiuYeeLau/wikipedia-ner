"""Simple test suite for the LSTMClassifier."""

import unittest
import numpy

from lstm import LSTMClassifier

class LSTMClassifierTest(unittest.TestCase):
    """Simple test suite for the LSTMClassifier."""
    def setUp(self):
        self.train_x = [
            # Sequence 1
            numpy.array([[1, 2, 1], [2, 1, 0], [2, 3, 1]]),
            # Sequence 2
            numpy.array([[1, 0, 0], [0, 0, 0]]),
            # Sequence 3
            numpy.array([[1, 0, 0], [2, 1, 0], [2, 3, 1]]),
        ]
        self.train_y = [
            [1, 2, 2],
            [1, 0],
            [1, 2, 2]
        ]
        self.classifier = LSTMClassifier(input_dim=3, lstm_state_size=2)

    def test_simple_case(self):
        self.classifier.fit(self.train_x, self.train_y)
        predictions = self.classifier.predict(self.train_x)
        self.assertEqual(3, len(predictions))
        self.assertEqual((3,), predictions[0].shape)



if __name__ == '__main__':
    unittest.main()