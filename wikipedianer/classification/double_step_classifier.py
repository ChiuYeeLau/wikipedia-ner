# -*- coding: utf-8 -*-
from __future__ import absolute_import
import logging
import numpy
import pandas
import pickle
import os

from scipy.sparse import csr_matrix, vstack

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tqdm import tqdm
from .mlp import MultilayerPerceptron
from .heuristic_classifier import HeuristicClassifier

logging.basicConfig(level=logging.INFO)


class ClassifierFactory(object):
    """Abstract class."""
    def get_classifier(self, dataset, experiment_name):
        raise NotImplementedError


class MLPFactory(ClassifierFactory):
    """"""
    def __init__(self, results_save_path, training_epochs, layers):
        self.results_save_path = results_save_path
        self.training_epochs = training_epochs
        self.layers = layers

    def get_classifier(self, dataset, experiment_name,
                       cl_iteration=1):
        batch_size = min(dataset.num_examples('train'), 2000,
                         dataset.num_examples('validation'))
        classifier = MultilayerPerceptron(
            dataset, results_save_path=self.results_save_path,
            experiment_name=experiment_name, layers=self.layers,
            save_model=True, cl_iteration=cl_iteration,
            batch_size=batch_size, training_epochs=self.training_epochs)
        return classifier


class HeuristicClassifierFactory(ClassifierFactory):
    def __init__(self, entities, features):
        self.entities = entities
        self.features = features

    def get_classifier(self, dataset, experiment_name):
        return HeuristicClassifier(dataset, self.entities, self.features)


class DoubleStepClassifier(object):
    """Double step classifier.
    The classification process has two stages:
        -- Use a trained classifier to determine a higher level category of an
            instance.
        -- Using the higher level category, select the corresponding low level
         classifier (entity linker) and use it to get the definitive label.

    This classifier takes three main pieces of information:
        -- The dataset and its split.
        -- The high level classifier already trained.
        -- The class of the low level classifier.
    """
    def __init__(self, dataset_class=None):
        """
        :param models_dirpath: string. The name of the directory where to store
            the trained models.
        :param negative_proportion: float < 1. The relation between negative and
            positive examples to use when constructing the datasets for training
             the low level classifiers.
        :param dataset_class: a subclass of dataset.Dataset
        """
        self.dataset = None

        self.hl_labels_name = None
        self.ll_labels_name = None

        self.dataset_class = dataset_class

        self.low_level_models = {}
        # Keeps the order assigned to the filtered class during the training
        # of the low level models
        self.low_level_classes_orders = {}

        self.test_results = {}

        # The accuracy is the total
        self.correctly_labeled = 0
        self.total_test_size = 0

    def load_from_files(self, dataset_filepath, labels_filepath,
                        labels, indices_filepath):
        """
        Builds the internal matrix from the given files.

        :param dataset_filepath:
        :param labels_filepath:
        :param labels: tuple with index and name of high level and low level
            class. The index is the position in the labels's tuples in filepath.
        :param indices_filepath:
        """
        self.dataset = self.dataset_class()
        self.dataset.load_from_files(
            dataset_filepath, labels_filepath, indices_filepath,
            cl_iterations=labels)
        self.classes = self.dataset.classes
        self.hl_labels_name = labels[0][1]
        self.ll_labels_name = labels[1][1]

    def load_from_arrays(self, x_matrix, hl_labels, ll_labels, train_indices,
                         test_indices, validation_indices,
                         hl_labels_name, ll_labels_name):
        """
        Builds the internal matrix from the given arrays.

        :param x_matrix: a 2-dimension sparse matrix with all examples.
        :param hl_labels: an array-like object with the high level classes.
        :param ll_labels: an array-like object with the low level classes.
        :param train_indices: an array-like object with the indices of instances
            of x_matrix to use for training.
        :param test_indices: an array-like object with the indices of instances
            of x_matrix to use for testing.
        :param validation_indices: an array-like object with the indices of
            instances of x_matrix to use for validation.
        """
        classes = [
            numpy.unique(hl_labels, return_inverse=True),
            numpy.unique(ll_labels, return_inverse=True)
        ]
        self.classes = tuple([cls[0] for cls in classes])
        integer_labels = numpy.stack([cls[1] for cls in classes]).T

        self.dataset = self.dataset_class()

        if len(test_indices):
            test_x = x_matrix[test_indices]
            test_labels = integer_labels[test_indices]
        else:
            test_x = csr_matrix([])
            test_labels = []
        if len(validation_indices):
            validation_x = x_matrix[validation_indices]
            validation_labels = integer_labels[
                validation_indices]
        else:
            validation_x = csr_matrix([])
            validation_labels = []

        self.dataset.load_from_arrays(
            self.classes, train_dataset=x_matrix[train_indices],
            test_dataset=test_x, validation_dataset=validation_x,
            train_labels=integer_labels[train_indices],
            test_labels=test_labels, validation_labels=validation_labels)
        self.ll_labels_name = ll_labels_name
        self.hl_labels_name = hl_labels_name

    def _filter_dataset(self, dataset_name, target_label):
        dataset = self.dataset.datasets[dataset_name]
        indices = numpy.where(dataset.labels[:, 0] == target_label)[0]

        return dataset.data[indices], dataset.labels[indices], indices

    def create_train_dataset(self, target_label_index):
        """
        Returns a numpy array with a subset of indices with balanced examples
        of target_label and negative examples, taken from labels.

        :param target_label_index: an integer. The index of the high level
            label associated with the train dataset.
        :return: a new instance of Dataset.
        """
        train_x, train_y, train_indices = self._filter_dataset(
            'train', target_label_index)
        test_x, test_y, test_indices = self._filter_dataset(
            'test', target_label_index)
        validation_x, validation_y, validation_indices = self._filter_dataset(
            'validation', target_label_index)

        logging.info('Creating dataset with sizes {} {} {} for {}'.format(
            train_x.shape[0], test_x.shape[0], validation_x.shape[0],
            self.classes[0][target_label_index]
        ))

        if (validation_x.shape[0] < 2 or test_x.shape[0] < 2 or
                    train_x.shape[0] < 2):
            logging.error('Dataset has less than 2 instances per split.')
            return None

        indices = {'train': train_indices, 'test': test_indices,
                   'validation': validation_indices}

        # Filter the classes based on the training dataset
        replaced_train_y = self.classes[1][train_y[:, 1]]
        replaced_test_y = self.classes[1][test_y[:, 1]]
        replaced_validation_y = self.classes[1][validation_y[:, 1]]
        all_labels = numpy.concatenate((replaced_train_y, replaced_test_y,
                                        replaced_validation_y))
        new_classes, int_labels = numpy.unique(all_labels, return_inverse=True)
        train_y[:,1] = int_labels[:train_y.shape[0]]
        test_y[:,1] = int_labels[:test_y.shape[0]]
        validation_y[:, 1] = int_labels[:validation_y.shape[0]]

        new_dataset = self.dataset_class()
        new_dataset.load_from_arrays(
            (self.classes[0], new_classes), train_x, test_x, validation_x,
            train_y, test_y, validation_y, indices)
        return new_dataset

    def train(self, classifier_factory):
        """Trains the classifier.

        :param low_level_classifier_class: python class. The class to
            instantiate when creating a low level classifier. Must extend
            base.BaseClassifier.
        :param classifier_factory: an function
        """
        # Train a different model for each hl_class
        for hl_label_index, hl_label in tqdm(enumerate(self.classes[0]),
                                             total=len(self.classes[0])):
            if hl_label == 'O':
                continue
            # Calculate indices for this high level class.
            new_dataset = self.create_train_dataset(hl_label_index)
            if not new_dataset:
                continue
            classifier = classifier_factory.get_classifier(
                new_dataset, experiment_name=hl_label)

            classifier.train(save_layers=False)
            self.low_level_models[hl_label] = classifier

            # We may need this to rebuild the classifiers.
            self.low_level_classes_orders[hl_label] = new_dataset.classes[1]
            self.test_results[hl_label] = classifier.test_results
            self.correctly_labeled += (
                new_dataset.num_examples('test') *
                classifier.test_results['accuracy'].max())
            self.total_test_size += new_dataset.num_examples('test')

    def _dataset_from_predictions(self, dataset_name, target_label_index,
                                  predicted_high_level_labels, new_labels):
        test_x, test_y, test_indices = self._filter_dataset(
            dataset_name, target_label_index)
        test_y[:, 1] = predicted_high_level_labels[test_indices]
        # Replace labels following the classes order
        old_labels = self.classes[1]
        new_labels_indexes = numpy.zeros(old_labels.shape) - 1
        for old_index, old_label in enumerate(old_labels):
            new_index = numpy.where(new_labels == old_label)[0]
            if new_index.shape[0] > 0:
                new_labels_indexes[old_index] = new_index
        numpy.place(
            test_y[:, 1], numpy.in1d(test_y, numpy.arange(old_labels.shape[0])),
            new_labels_indexes.astype(numpy.int32))
        return test_x, test_y

    def predict(self, dataset_name, predicted_high_level_labels=None,
                default_label='O'):
        if default_label in self.classes[1]:
            default_index = numpy.where(self.classes[1] == default_label)[0][0]
        else:
            default_index = 0
        predictions = (numpy.zeros((self.dataset.num_examples(dataset_name),)) +
                       default_index)
        if not self.dataset:
            raise ValueError('A dataset must be loaded previously')
        if isinstance(predicted_high_level_labels, list):
            predicted_high_level_labels = numpy.array(
                predicted_high_level_labels)
        for hl_label_index, hl_label in tqdm(enumerate(self.classes[0]),
                                             total=len(self.classes[0])):
            if not hl_label in self.low_level_models:
                continue
            model = self.low_level_models[hl_label]
            if predicted_high_level_labels is None:
                results = model.evaluate(dataset_name, return_extras=True)
            else:
                # Replace the dataset_name with a new one filtered from
                # the given high level predictions
                original_test_dataset = model.dataset.datasets[dataset_name]
                test_x, test_y = self._dataset_from_predictions(
                    dataset_name, hl_label, predicted_high_level_labels,
                    model.dataset.classes[1])
                model.dataset.add_dataset(dataset_name, test_x, test_y)
                results = model.evaluate(dataset_name, return_extras=True)
                model.dataset.add_dataset(
                    dataset_name, original_test_dataset.data,
                    original_test_dataset.labels)
            y_true, y_pred = results[-2:]
            # Now we need to convert from low level dataset labels to the
            # high level labels.
            y_pred = model.dataset.classes[1][y_pred]
            test_indices = model.dataset.indices[dataset_name]
            for index, low_level_class in enumerate(self.classes[1]):
                low_level_indices = numpy.where(
                    y_pred == low_level_class)[0]
                predictions[test_indices[low_level_indices]] = index
        return predictions

    def evaluate(self, predicted_high_level_labels=None,
                 classifier_factory=None, default_label='O'):
        """Evalutes the classifier in a real pipeline over the test dataset.

        Uses the predicted_high_level_labels to select a low level classifier
        to apply to the instance.
        """
        y_true = self.dataset.datasets['test'].labels[:,1]
        y_pred = self.predict('test', predicted_high_level_labels,
                              default_label)
        accuracy = accuracy_score(y_true, y_pred.astype(y_true.dtype))

        labels = numpy.arange(self.dataset.output_size(1))
        precision = precision_score(y_true, y_pred, labels=labels, average=None)
        recall = recall_score(y_true, y_pred, labels=labels, average=None)
        fscore = f1_score(y_true, y_pred, labels=labels, average=None)

        return accuracy, precision, recall, fscore, y_true, y_pred


    def save_to_file(self, results_dirname):
        """Saves classifier metadata and test results to files"""
        to_save = {
            'classes_orders': self.low_level_classes_orders,
            'classes': self.classes
        }
        filename = os.path.join(results_dirname,
                                'double_step_classifier.meta.pickle')
        with open(filename, 'wb') as output_file:
            pickle.dump(to_save, output_file)

        filename = os.path.join(results_dirname,
                                'general_test_results.csv')
        results = pandas.concat(self.test_results.values())
        general_accuracy = self.correctly_labeled / self.total_test_size
        results = results.append({'general_accuracy': general_accuracy},
                                 ignore_index=True)
        results.to_csv(filename, index=False)