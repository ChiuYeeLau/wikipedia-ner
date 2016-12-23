"""Test for the DoubleStepClassifier and ClassifierFactory classes."""

import numpy
import unittest

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch
from scipy.sparse import csr_matrix
from wikipedianer.dataset import HandcraftedFeaturesDataset
from wikipedianer.pipeline import util
from wikipedianer.classification.double_step_classifier import (
        DoubleStepClassifier, MLPFactory)


class DoubleStepClassifierTest(unittest.TestCase):

    def setUp(self):
        x_matrix = csr_matrix([
            [0, 1, 0], [0, 0, 0], [0, 4, 0], [1, 1, 1], [1, 0, 0],
            [1, 0, 1], [0, 2, 0], [0, 1, 0],
            [0, 2, 1], [0, 3, 0], [1, 3, 1], [0, 2, 0]
        ])
        train_indices = [0, 1, 2, 3, 4]
        test_indices = [5, 6, 7]
        validation_indices = [8, 9, 10, 11]
        # hl label is first element, ll label is last element
        hl_labels = [
            '0', '0', '0', '1', '1',
            '1', '0', '0',
            '0', '1', '1', '0'
        ]
        ll_labels = [
            '00', '01', '01', '11', '10',
            '11', '00', '01',
            '01', '00', '11', '00'
        ]
        hl_labels_name = util.CL_ITERATIONS[-2]
        ll_labels_name = util.CL_ITERATIONS[-1]

        self.classifier = DoubleStepClassifier(
            dataset_class=HandcraftedFeaturesDataset)
        self.classifier.load_from_arrays(
            x_matrix, hl_labels, ll_labels, train_indices,
            test_indices, validation_indices, hl_labels_name,
            ll_labels_name)

    @patch('wikipedianer.classification.double_step_classifier.MultilayerPerceptron.save_model')
    @patch('wikipedianer.classification.double_step_classifier.MultilayerPerceptron._save_results')
    def test_basic_train(self, save_model_mock, save_results_mock):
        """Test the training of with a simple matrix."""
        classifier_factory = MLPFactory('', 10, [10])
        self.classifier.train(classifier_factory)

        # One of the datasets is too small
        self.assertEqual(1, len(self.classifier.test_results))

    @patch('wikipedianer.classification.double_step_classifier.'
           'MultilayerPerceptron.save_model')
    @patch('wikipedianer.classification.double_step_classifier'
           '.MultilayerPerceptron._save_results')
    def test_basic_evaluate(self, save_model_mock, save_results_mock):
        """Test the training of with a simple matrix."""
        classifier_factory = MLPFactory('', 10, [10])
        self.classifier.train(classifier_factory)
        # Evaluates the classifier without loading from files.
        self.classifier.evaluate([0, 0, 1], classifier_factory)

    def test_create_dataset(self):
        result_dataset = self.classifier.create_train_dataset(0)
        self.assertIsNotNone(result_dataset)
        self.assertEqual(3, result_dataset.num_examples('train'))
        self.assertEqual(2, result_dataset.num_examples('test'))
        self.assertEqual(2, result_dataset.num_examples('validation'))

        labels = result_dataset.dataset_labels(dataset_name='train',
                                               cl_iteration=1)
        self.assertEqual(numpy.unique(labels).shape[0], 2)

        labels = result_dataset.dataset_labels(dataset_name='validation',
                                               cl_iteration=1)
        self.assertEqual(numpy.unique(labels).shape[0], 2)

        # Check the total number of labels is correct.
        self.assertEqual(len(result_dataset.classes[1]), 2)
        train_labels = result_dataset.classes[1][
            result_dataset.datasets['train'].labels[:,1]
        ]
        self.assertEqual(['00', '01'],
                         sorted(numpy.unique(train_labels.tolist())))

    def test_create_dataset_small(self):
        result_dataset = self.classifier.create_train_dataset(1)
        self.assertIsNone(result_dataset)


if __name__ == '__main__':
    unittest.main()
