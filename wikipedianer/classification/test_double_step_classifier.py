"""Test for the DoubleStepClassifier and ClassifierFactory classes.

Recommended run as

$ script -c '~/anaconda2/envs/env35/bin/nosetests -s wikipedianer/classification/test_double_step_classifier.py' | grep -v 'Level 1'

to delete tensorflow logs.
"""

import numpy
import os
import shutil
import tensorflow as tf

import unittest

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch
from scipy.sparse import csr_matrix
from wikipedianer.dataset import HandcraftedFeaturesDataset
from wikipedianer.dataset.preprocess import StratifiedSplitter
from wikipedianer.pipeline import util
from wikipedianer.classification.base import BaseClassifier
from wikipedianer.classification.double_step_classifier import (
        DoubleStepClassifier, MLPFactory, ClassifierFactory)
from wikipedianer.classification.logistic_regression import (
    LRClassifierFactory
)


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


class MockClassifier(BaseClassifier):
    def __init__(self, dataset):
        super(MockClassifier, self).__init__()
        self.dataset = dataset

    def train(self, *args, **kwargs):
        accuracy = 0
        prec = numpy.array([0.5 for label in self.dataset.classes[1]])
        recall = numpy.array([0.4 for label in self.dataset.classes[1]])
        f1 = numpy.array([0.6 for label in self.dataset.classes[1]])
        self.add_test_results(accuracy, prec, recall, f1,
                              classes=self.dataset.classes[1])

    def evaluate(self, dataset_name, restore=False, *args, **kwargs):
        # All these are random numbers that doesn't matter
        accuracy = 0
        prec = numpy.array([0.5 for label in self.dataset.classes[1]])
        recall = numpy.array([0.4 for label in self.dataset.classes[1]])
        f1 = numpy.array([0.6 for label in self.dataset.classes[1]])
        # Always return the first class
        self.selected_class = self.dataset.classes[1][0]
        predictions = numpy.zeros(self.dataset.num_examples(dataset_name),
                                  dtype=numpy.int32)
        return (accuracy, prec, recall, f1,
            self.dataset.datasets[dataset_name].labels, predictions)


class MockClassifierFactory(ClassifierFactory):
    def get_classifier(self, dataset, experiment_name):
        return MockClassifier(dataset)


class DoubleStepMockClassifierTest(unittest.TestCase):

    def setUp(self):
        self.num_examples = 200
        self.feature_size = 5
        x_matrix = csr_matrix(
            numpy.random.randint(0, 10, size=(self.num_examples,
                                              self.feature_size)))
        high_level_labels = numpy.random.randint(
            0, 10, size=(self.num_examples,))
        low_level_labels = numpy.random.randint(
            0, 10, size=(self.num_examples,))
        low_level_labels = (high_level_labels * 10 + low_level_labels).astype(
            numpy.str)
        high_level_labels = high_level_labels.astype(numpy.str)
        train_size = int(self.num_examples * 0.7)
        test_size = int(self.num_examples * 0.2)
        self.classifier = DoubleStepClassifier(HandcraftedFeaturesDataset)
        self.classifier.load_from_arrays(
            x_matrix, hl_labels=high_level_labels, ll_labels=low_level_labels,
            train_indices=range(train_size),
            test_indices=range(train_size, train_size+test_size),
            validation_indices=range(train_size+test_size, self.num_examples),
            hl_labels_name=util.CL_ITERATIONS[-2],
            ll_labels_name=util.CL_ITERATIONS[-1])

    def test_evaluate(self):
        factory = MockClassifierFactory()
        self.classifier.train(factory)
        results = self.classifier.evaluate(classifier_factory=factory)
        predictions = results[-1]
        self.assertEqual(self.classifier.dataset.num_examples('test'),
                         predictions.shape[0])

        predictions = self.classifier.classes[1][predictions]
        trained_hl_labels = sorted(self.classifier.low_level_models.keys())
        true_hl_labels = self.classifier.dataset.datasets['test'].labels[:,0]
        expected_predictions = numpy.array(
            [self.classifier.classes[1][0]] * predictions.shape[0]).astype(
            self.classifier.classes[1].dtype)
        for hl_label_index, hl_label in enumerate(self.classifier.classes[0]):
            if hl_label in trained_hl_labels:
                selected_prediction = self.classifier.low_level_models[
                    hl_label][0].selected_class
                mask = numpy.where(true_hl_labels == hl_label_index)
                expected_predictions[mask] = selected_prediction

        self.assertTrue(numpy.array_equal(expected_predictions, predictions))


def get_model_variable_values(model, restore=False, session=None):
    values = {}
    if restore:
        session = tf.Session()
        model.saver.restore(session, model._get_save_path())
    for var_name, tensor in model.var_names.items():
        tensor_value = session.run(tensor)
        values[var_name] = tensor_value
    return values


class ClassifierFactoryTest(unittest.TestCase):

    OUTPUT_DIR = os.path.join('wikipedianer', 'classification', 'test_files')

    def setUp(self):
        self.num_examples = 100
        self.feature_size = 5
        x_matrix = csr_matrix(
            numpy.random.randint(0, 2, size=(self.num_examples,
                                             self.feature_size)))
        hl_labels_num = 10
        ll_labels_num = 10
        classes = (numpy.arange(0, hl_labels_num).astype(numpy.str),
                   numpy.arange(0, ll_labels_num).astype(numpy.str))
        hl_labels = numpy.random.randint(
            0, 10, size=(self.num_examples,))
        ll_labels = numpy.random.randint(
            0, 20, size=(self.num_examples,))

        classes = [
            numpy.unique(hl_labels, return_inverse=True),
            numpy.unique(ll_labels, return_inverse=True)
        ]
        self.classes = tuple([cls[0] for cls in classes])
        integer_labels = numpy.stack([cls[1] for cls in classes]).T

        self.dataset = HandcraftedFeaturesDataset()
        train_size = int(self.num_examples * 0.7)
        test_size = int(self.num_examples * 0.2)
        self.dataset.load_from_arrays(
            self.classes, train_dataset=x_matrix[:train_size,:],
            test_dataset=x_matrix[train_size:train_size + test_size,:],
            validation_dataset=x_matrix[train_size + test_size:,:],
            train_labels=integer_labels[:train_size,:],
            test_labels=integer_labels[train_size:train_size + test_size,:],
            validation_labels=integer_labels[train_size + test_size:,:])
        safe_mkdir(self.OUTPUT_DIR)

    def tearDown(self):
        safe_rmdir(self.OUTPUT_DIR)

    def _compare_values(self, values1, values2):
        for var_name, tensor in values1.items():
            self.assertTrue(
                numpy.array_equal(tensor, values2[var_name]))

    @patch('wikipedianer.classification.double_step_classifier'
           '.MultilayerPerceptron._save_results')
    def test_save_and_load_classifier(self, save_results_mock):
        factory = MLPFactory(results_save_path=self.OUTPUT_DIR)
        model = factory.get_classifier(self.dataset,
                                       experiment_name='experiment')
        session = model.train(save_layers=False, close_session=False)
        values = get_model_variable_values(model, session=session)
        del model

        model = factory.get_classifier(self.dataset,
                                       experiment_name='experiment')
        new_values = get_model_variable_values(model, restore=True)

        self._compare_values(values1=values, values2=new_values)

    @patch('wikipedianer.classification.double_step_classifier'
           '.MultilayerPerceptron._save_results')
    def test_save_and_load_multiple_classifier(self, save_results_mock):
        factory = MLPFactory(results_save_path=self.OUTPUT_DIR,
                             training_epochs=10)
        model = factory.get_classifier(self.dataset,
                                       experiment_name='experiment1')
        session = model.train(save_layers=False, close_session=False)
        values1 = get_model_variable_values(model, session=session)
        del model
        model = factory.get_classifier(self.dataset,
                                       experiment_name='experiment2')
        session = model.train(save_layers=False, close_session=False)
        values2 = get_model_variable_values(model, session=session)
        del model

        model = factory.get_classifier(self.dataset,
                                       experiment_name='experiment1')
        new_values1 = get_model_variable_values(model, restore=True)
        del model
        model = factory.get_classifier(self.dataset,
                                       experiment_name='experiment2')
        new_values2 = get_model_variable_values(model, restore=True)
        self._compare_values(values1=values1, values2=new_values1)
        self._compare_values(values1=values2, values2=new_values2)


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
        test_dataset = self.classifier.low_level_models[
            '0'][0].dataset.datasets['test']
        predictions = self.classifier.predict(
            'test', predicted_high_level_labels=[1, 1, 0],
            default_label=4).tolist()
        self.assertEqual(len(predictions), 3)

        self.assertEqual(predictions[0], 0)
        self.assertEqual(predictions[1], 0)  # For non trained classifier is
        # default index
        self.assertNotEqual(self.classifier.classes[1][predictions[1]], '11')

        new_test_dataset = self.classifier.low_level_models[
            '0'][0].dataset.datasets['test']
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
            results_save_path=self.OUTPUT_DIR, training_epochs=10)
        self.classifier.train(classifier_factory)
        original_results = self.classifier.evaluate()
        original_test, original_predictions = original_results[-2:]
        # Delete the reference to the model, forcing to read from file.
        self.classifier.close_open_sessions()
        new_results = self.classifier.evaluate(
            classifier_factory=classifier_factory)
        new_test, new_predictions = new_results[-2:]
        # Evaluates the classifier without loading from files.
        self.assertEqual(original_test.tolist(), new_test.tolist())
        self.assertEqual(original_predictions.tolist(),
                         new_predictions.tolist())
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

    def test_predict_lr(self):
        """Test the predict function."""
        classifier_factory = LRClassifierFactory()
        self.classifier.train(classifier_factory)
        predictions = self.classifier.predict('test', default_label=4).tolist()
        self.assertEqual(len(predictions), 3)

        self.assertEqual(predictions[0], 0)  # For non trained classifier is
        # default index
        self.assertNotEqual(self.classifier.classes[1][predictions[1]], '11')
        self.assertNotEqual(self.classifier.classes[1][predictions[2]], '11')

    def test_evaluate_with_files_lr(self):
        """Test the evaluation saving the models to a file."""
        safe_rmdir(self.OUTPUT_DIR)
        safe_mkdir(self.OUTPUT_DIR)
        classifier_factory = LRClassifierFactory(
            results_save_path=self.OUTPUT_DIR, save_models=True)
        self.classifier.train(classifier_factory)
        original_results = self.classifier.evaluate()
        original_test, original_predictions = original_results[-2:]
        # Delete the reference to the model, forcing to read from file.
        self.classifier.close_open_sessions()
        new_results = self.classifier.evaluate(
            classifier_factory=classifier_factory)
        new_test, new_predictions = new_results[-2:]
        # Evaluates the classifier without loading from files.
        self.assertEqual(original_test.tolist(), new_test.tolist())
        self.assertEqual(original_predictions.tolist(),
                         new_predictions.tolist())
        safe_rmdir(self.OUTPUT_DIR)

    def test_predict_w_given_labels_lr(self):
        """Test the evaluation of with a simple matrix."""
        classifier_factory = LRClassifierFactory()
        self.classifier.train(classifier_factory)
        test_dataset = self.classifier.low_level_models[
            '0'][0].dataset.datasets['test']
        predictions = self.classifier.predict(
            'test', predicted_high_level_labels=[1, 1, 0],
            default_label=4).tolist()
        self.assertEqual(len(predictions), 3)

        self.assertEqual(predictions[0], 0)
        self.assertEqual(predictions[1], 0)  # For non trained classifier is
        # default index
        self.assertNotEqual(self.classifier.classes[1][predictions[1]], '11')

        new_test_dataset = self.classifier.low_level_models[
            '0'][0].dataset.datasets['test']
        self.assertTrue(numpy.array_equal(test_dataset.data,
                                          new_test_dataset.data))
        self.assertTrue(numpy.array_equal(test_dataset.labels,
                                          new_test_dataset.labels))

if __name__ == '__main__':
    unittest.main()
