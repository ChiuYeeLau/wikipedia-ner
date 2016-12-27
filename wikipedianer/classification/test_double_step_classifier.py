"""Test for the DoubleStepClassifier and ClassifierFactory classes."""

import numpy
import os
import shutil
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


def safe_mkdir(dir_path):
    """Checks if a directory exists, and if it doesn't, creates one."""
    try:
        os.stat(dir_path)
    except OSError:
        os.mkdir(dir_path)


def safe_rmdir(dir_path):
    """Checks if a directory exists, and if it does, removes it."""
    try:
        os.stat(dir_path)
        shutil.rmtree(dir_path)
    except OSError:
        pass


class DoubleStepClassifierTest(unittest.TestCase):

    OUTPUT_DIR = os.path.join('wikipedianer', 'classification', 'test_files')

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

    @patch('wikipedianer.classification.double_step_classifier.'
           'MultilayerPerceptron.save_model')
    @patch('wikipedianer.classification.double_step_classifier'
           '.MultilayerPerceptron._save_results')
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
    def test_predict(self, save_model_mock, save_results_mock):
        """Test the predict function."""
        classifier_factory = MLPFactory('', 10, [10])
        self.classifier.train(classifier_factory)
        predictions = self.classifier.predict('test', default_label=4).tolist()
        self.assertEqual(len(predictions), 3)

        self.assertEqual(predictions[0], 0)  # For non trained classifier is
        # default index
        self.assertNotEqual(self.classifier.classes[1][predictions[1]], '11')
        self.assertNotEqual(self.classifier.classes[1][predictions[2]], '11')

    @patch('wikipedianer.classification.double_step_classifier.'
           'MultilayerPerceptron.save_model')
    @patch('wikipedianer.classification.double_step_classifier'
           '.MultilayerPerceptron._save_results')
    def test_predict_w_given_labels(self, save_model_mock, save_results_mock):
        """Test the evaluation of with a simple matrix."""
        classifier_factory = MLPFactory('', 10, [10])
        self.classifier.train(classifier_factory)
        test_dataset = self.classifier.low_level_models['0'].dataset.datasets[
            'test']
        predictions = self.classifier.predict(
            'test', predicted_high_level_labels=[1, 1, 0],
            default_label=4).tolist()
        self.assertEqual(len(predictions), 3)

        self.assertEqual(predictions[0], 0)
        self.assertEqual(predictions[1], 0)  # For non trained classifier is
        # default index
        self.assertNotEqual(self.classifier.classes[1][predictions[1]], '11')

        new_test_dataset = self.classifier.low_level_models[
            '0'].dataset.datasets['test']
        self.assertTrue(numpy.array_equal(test_dataset.data,
                                          new_test_dataset.data))
        self.assertTrue(numpy.array_equal(test_dataset.labels,
                                          new_test_dataset.labels))

    @patch('wikipedianer.classification.double_step_classifier.'
           'MultilayerPerceptron.save_model')
    @patch('wikipedianer.classification.double_step_classifier'
           '.MultilayerPerceptron._save_results')
    def test_basic_evaluate(self, save_model_mock, save_results_mock):
        """Test the evaluation of with a simple matrix."""
        classifier_factory = MLPFactory('', 10, [10])
        self.classifier.train(classifier_factory)
        # Evaluates the classifier without loading from files.
        results = self.classifier.evaluate()
        self.assertEqual(len(results), 6)
        accuracy = results[0]
        self.assertLessEqual(accuracy, 1)
        self.assertGreaterEqual(accuracy, 0)
        for result in results[1:4]:
            for metric in result:
                self.assertLessEqual(metric, 1)
                self.assertGreaterEqual(metric, 0)

    @patch('wikipedianer.classification.double_step_classifier'
           '.MultilayerPerceptron._save_results')
    def test_evaluate_with_files(self, save_results_mock):
        """Test the evaluation saving the models to a file."""
        safe_rmdir(self.OUTPUT_DIR)
        safe_mkdir(self.OUTPUT_DIR)
        classifier_factory = MLPFactory(
            results_save_path=self.OUTPUT_DIR, training_epochs=10,
            layers=[10])
        self.classifier.train(classifier_factory)
        original_results = self.classifier.evaluate()
        original_test, original_predictions = original_results[-2:]
        # Delete the reference to the model, forcing to read from file.
        self.classifier.low_level_models = {}
        new_results = self.classifier.evaluate(
            classifier_factory=classifier_factory)
        new_test, new_predictions = new_results[-2:]
        # Evaluates the classifier without loading from files.
        self.assertEqual(original_test.tolist(), new_test.tolist())
        self.assertEqual(original_predictions.tolist(), new_predictions.tolist())
        safe_rmdir(self.OUTPUT_DIR)

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
