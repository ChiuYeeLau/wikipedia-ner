"""Simple test suite for the LSTMClassifier.
"""

import unittest
import numpy

from lstm import LSTMClassifier, SequentialPipeline


class SequentialPipelineTest(unittest.TestCase):

    def setUp(self):
        self.x_matrix = [
            # Sequence 1
            numpy.array([[1, 2, 1], [2, 1, 0], [2, 3, 1]]),
            # Sequence 2
            numpy.array([[1, 0, 0], [0, 0, 0]]),
            # Sequence 3
            numpy.array([[1, 0, 0]]),
        ]
        self.y_vector = [
            numpy.array([1, 2, 2]),
            numpy.array([1, 0]),
            numpy.array([1])
        ]

    def test_padding(self):
        result = SequentialPipeline.pad_sequences(self.x_matrix,
                                                  sequence_max_length=4)
        self.assertEqual((3, 4, 3), result.shape)
        self.assertEqual([0.0, 0.0, 0.0], result[1,2,:].tolist())

    def test_padding_smaller_shape(self):
        result = SequentialPipeline.pad_sequences(self.x_matrix,
                                                  sequence_max_length=2)
        self.assertEqual((3, 2, 3), result.shape)

    def test_padding_label_vector(self):
        result = SequentialPipeline.pad_sequences(self.y_vector,
                                                  sequence_max_length=3)
        self.assertEqual((3, 3), result.shape)


class LSTMClassifierTest(unittest.TestCase):
    """Simple test suite for the LSTMClassifier."""
    def setUp(self):
        self.train_x = SequentialPipeline.pad_sequences([
            # Sequence 1
            numpy.array([[1, 2, 1], [2, 1, 0], [2, 3, 1]]),
            # Sequence 2
            numpy.array([[1, 0, 0], [0, 0, 0]]),
            # Sequence 3
            numpy.array([[1, 0, 0]]),
        ], sequence_max_length=3)
        self.train_y = SequentialPipeline.pad_sequences([
            numpy.array([1, 2, 2]),
            numpy.array([1, 0]),
            numpy.array([1])
        ], sequence_max_length=3)

    def test_simple_case(self):

        self.classifier = LSTMClassifier()
        self.classifier.fit(self.train_x, self.train_y)
        predictions = self.classifier.predict(self.train_x)
        self.assertEqual(3, len(predictions))
        self.assertEqual((3,), predictions[0].shape)


if __name__ == '__main__':
    unittest.main()