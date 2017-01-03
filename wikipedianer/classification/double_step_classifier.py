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

    def get_results_filename(self, experiment_name):
        raise NotImplementedError


class MLPFactory(ClassifierFactory):
    """"""
    def __init__(self, results_save_path, training_epochs=1000,
                 dropout_ratio=0.0, num_layers=1):
        self.results_save_path = results_save_path
        self.training_epochs = training_epochs
        self.dropout_ratio = dropout_ratio
        self.num_layers = min(num_layers, 1)

    def get_classifier(self, dataset, experiment_name,
                       cl_iteration=1):
        batch_size = min(dataset.num_examples('train'), 2000,
                         dataset.num_examples('validation'),
                         dataset.num_examples('test'))
        if self.num_layers == 1:
            # The hidden layer is twice the size of the number of labels, with
            # a maximum of 500 and a minimum of 10.
            num_labels = dataset.classes[1].shape[0]
            hidden_layers = [min(max(10, 2 * num_labels), 500)]
        else:
            assert self.num_layers == 0
            hidden_layers = []

        loss_report = min(batch_size, 250)

        classifier = MultilayerPerceptron(
            dataset, results_save_path=self.results_save_path,
            experiment_name=experiment_name, layers=hidden_layers,
            save_model=True, cl_iteration=cl_iteration,
            batch_size=batch_size, training_epochs=self.training_epochs,
            loss_report=loss_report, dropout_ratios=[self.dropout_ratio])
        return classifier

    def get_results_filename(self, experiment_name):
        return os.path.join(self.results_save_path,
                            'test_results_%s.csv' % experiment_name)


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
    def __init__(self, dataset_class=None, use_trained=False):
        """
        :param models_dirpath: string. The name of the directory where to store
            the trained models.
        :param negative_proportion: float < 1. The relation between negative and
            positive examples to use when constructing the datasets for training
             the low level classifiers.
        :param dataset_class: a subclass of dataset.Dataset
        """
        self.dataset = None
        self.use_trained = use_trained

        self.hl_labels_name = None
        self.ll_labels_name = None

        self.dataset_class = dataset_class

        self.low_level_models = {}

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

            test_results = None
            if self.use_trained:
                # Read the test results from disk
                try:
                    filename = classifier_factory.get_results_filename(hl_label)
                    test_results = pandas.read_csv(filename)
                    logging.info('Reading old results{}'.format(filename))
                except Exception:
                    pass

            if test_results is None:
                try:
                    classifier = classifier_factory.get_classifier(
                        new_dataset, experiment_name=hl_label)
                except Exception as e:
                    logging.error('Classifier {} not created. Error {}'.format(
                        hl_label, e))
                    continue
                logging.info('Training classifier {}'.format(hl_label))
                session = classifier.train(save_layers=False)
                test_results = classifier.test_results
                # self.low_level_models[hl_label] = (classifier, session)

            self.test_results[hl_label] = test_results
            self.test_results[hl_label]['hl_label'] = hl_label
            self.correctly_labeled += (
                new_dataset.num_examples('test') *
                test_results['accuracy'].max())
            self.total_test_size += new_dataset.num_examples('test')

    def _dataset_from_predictions(self, dataset_name, target_label_index,
                                  predicted_high_level_labels, new_labels):
        test_indices = numpy.where(
            predicted_high_level_labels == target_label_index)
        dataset_to_filter = self.dataset.datasets[dataset_name]
        test_x = dataset_to_filter.data[test_indices]
        test_y = dataset_to_filter.labels[test_indices]

        # Replace low level labels to match the classifier order.

        old_labels = self.classes[1]
        new_test_y = numpy.zeros(test_y.shape) - 1
        for old_index, old_label in enumerate(old_labels):
            # We search for the new index of the label "old_label"
            new_index = numpy.where(new_labels == old_label)[0]
            # old_label may not be present in new_labels.
            if new_index.shape[0] > 0:
                mask = numpy.where(test_y[:,1] == old_index)[0]
                new_test_y[mask, 1] = new_index
        return test_x, test_y, test_indices[0]

    def _predict_for_label(self, hl_label_index, classifier_factory,
                           predicted_high_level_labels, dataset_name,
                           predictions):
        """Completes the predictions array with the classifier results
        for hl_label_index
        """
        load_model = False
        session = None
        hl_label = self.classes[0][hl_label_index]
        if not hl_label in self.low_level_models:
            if classifier_factory is None:
                return
            # Read the model from file
            new_dataset = self.create_train_dataset(hl_label_index)
            # numpy.unique returns sorted arrays, which guaranties that every
            # time that we call create_train_dataset with the same parameters
            # we obtain the same result and the same order of classes.
            if not new_dataset:
                logging.warning('Evaluation dataset could not be created.')
                return
            try:
                model = classifier_factory.get_classifier(new_dataset, hl_label)
            except Exception as e:
                logging.error('Classifier {} not created. Error {}'.format(
                    hl_label, e))
                return
            load_model = True
            self.low_level_models[hl_label] = (model, None)
        else:
            model, session = self.low_level_models[hl_label]

        if predicted_high_level_labels is None:
            results = model.evaluate(dataset_name, return_extras=True,
                                     restore=load_model, session=session)
            test_indices = model.dataset.indices[dataset_name]
        else:
            assert (predicted_high_level_labels.shape[0] ==
                    self.dataset.num_examples(dataset_name))
            # Replace the dataset_name with a new one filtered from
            # the instances with hl_label_index in predicted_high_level_labels
            original_test_dataset = model.dataset.datasets[dataset_name]
            test_x, test_y, test_indices = self._dataset_from_predictions(
                dataset_name, hl_label_index, predicted_high_level_labels,
                model.dataset.classes[1])
            assert test_y.shape[0] == test_x.shape[0]
            assert test_y.shape[0] == test_indices.shape[0]
            model.dataset.add_dataset(dataset_name, test_x, test_y)
            results = model.evaluate(dataset_name, return_extras=True,
                                     restore=load_model, session=session)
            model.dataset.add_dataset(
                dataset_name, original_test_dataset.data,
                original_test_dataset.labels)
        y_true, y_pred = results[-2:]
        # Now we need to convert from low level dataset labels to the
        # high level labels.
        y_pred = model.dataset.classes[1][y_pred]
        for index, low_level_class in enumerate(self.classes[1]):
            low_level_indices = numpy.where(
                y_pred == low_level_class)[0]
            predictions[test_indices[low_level_indices]] = index

    def predict(self, dataset_name, predicted_high_level_labels=None,
                classifier_factory=None, default_label='O'):
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
            if hl_label == default_label:
                continue
            self._predict_for_label(
                hl_label_index, classifier_factory, predicted_high_level_labels,
                dataset_name, predictions)
        return predictions.astype(numpy.int32)

    def evaluate(self, predicted_high_level_labels=None,
                 classifier_factory=None, default_label='O'):
        """Evalutes the classifier in a real pipeline over the test dataset.

        Uses the predicted_high_level_labels to select a low level classifier
        to apply to the instance.
        """
        y_true = self.dataset.datasets['test'].labels[:,1]
        y_pred = self.predict('test', predicted_high_level_labels,
                              classifier_factory, default_label)
        accuracy = accuracy_score(y_true, y_pred.astype(y_true.dtype))

        labels = numpy.arange(self.dataset.output_size(1))
        precision = precision_score(y_true, y_pred, labels=labels, average=None)
        recall = recall_score(y_true, y_pred, labels=labels, average=None)
        fscore = f1_score(y_true, y_pred, labels=labels, average=None)

        return accuracy, precision, recall, fscore, y_true, y_pred

    def close_open_sessions(self):
        for _, session in self.low_level_models.values():
            if session is not None:
                session.close()
        self.low_level_models = {}

    def save_to_file(self, results_dirname):
        """Saves classifier metadata and test results to files"""
        to_save = {
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
