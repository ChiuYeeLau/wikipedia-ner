"""A Nearest Neighbors classifier using only bigram and trigram information."""
import pickle

import logging

import pandas
from sklearn import neighbors

import numpy
import os

from .base import BaseClassifier
from .double_step_classifier import ClassifierFactory


def get_ngram_set(token, prev, next_):
    return set([(token, ), (prev, token), (token, next_), (prev, token, next_)])


class NNeighborsClassifierFactory(ClassifierFactory):
    def __init__(self, features_filename, results_save_path):
        # Read the features
        with open(features_filename, 'rb') as feature_file:
            features = pickle.load(feature_file)
            # Get indices for each current token
        token_features = [index for index, x in enumerate(features)
                          if x.startswith('token:current')]
        prev_features = [index for index, x in enumerate(features)
                         if x.startswith('token:prev')]
        next_features = [index for index, x in enumerate(features)
                         if x.startswith('token:next')]
        self.valid_features = numpy.concatenate([token_features, next_features,
                                                 prev_features])
        self.total_features = len(self.valid_features)
        self.results_save_path = results_save_path

    def get_classifier(self, dataset, experiment_name):
        return NNeighborsClassifier(
            dataset, self.valid_features, self.results_save_path,
            experiment_name=experiment_name)


class NNeighborsClassifier(BaseClassifier):

    DEFAULT_LABEL = 'O'

    def __init__(self, dataset, valid_features, results_save_path,
                 experiment_name='', n_neighbors=3):
        """
        :param dataset: A Dataset instance
        :param features: A dictionary from indexes in the feature array to
         tokens. Only indices of features indicating the current token are used.
        """
        super(NNeighborsClassifier, self).__init__()
        self.dataset = dataset
        self.valid_features = valid_features
        self.classifier = neighbors.KNeighborsClassifier(n_neighbors,
                                                         weights='distance')
        self.results_save_path = results_save_path
        self.experiment_name = experiment_name
        self.trained = False

    def predict(self, dataset_name):
        x_matrix = self.dataset.datasets[dataset_name].data[
                   :, self.valid_features]
        y_pred = numpy.zeros(self.dataset.num_examples(dataset_name),
                             dtype=numpy.int32)
        batch_size = min(self.dataset.num_examples(dataset_name), 2000)

        for step, dataset_chunk in self.dataset.traverse_dataset(
                dataset_name, batch_size):
            end = min(step + batch_size,
                      self.dataset.num_examples(dataset_name))
            y_pred[step:end] = self.classifier.predict(x_matrix[step:end])
        return y_pred

    def evaluate(self, dataset_name='test', restore=False, *args, **kwargs):
        if restore and not self.trained:
            self.read()
        logging.info('Obtaining predictions')
        y_pred = self.predict(dataset_name)
        y_true = self.dataset.dataset_labels(dataset_name, 1)
        assert y_pred.shape[0] == y_true.shape[0]
        return self.get_metrics(y_true, y_pred, return_extras=True)

    def train(self, save_layers=False):
        self.classifier.fit(
            self.dataset.datasets['train'].data[:, self.valid_features],
            self.dataset.dataset_labels('train', 1)
        )
        self.trained = True
        logging.info('Training completed.')

        accuracy, precision, recall, fscore, y_true, y_pred = self.evaluate()

        self.add_test_results(accuracy, precision, recall, fscore,
                              classes=self.dataset.classes[1], y_true=y_true)
        self.save()
        self.test_results.to_csv(self.get_results_filename(), index=False)

    def get_save_filename(self):
        return os.path.join(self.results_save_path,
                            '{}_knn.model'.format(self.experiment_name))

    def get_results_filename(self):
        return os.path.join(
            self.results_save_path,
            'test_results_{}.csv'.format(self.experiment_name))

    def save(self):
        logging.info('Saving model to file')
        with open(self.get_save_filename(), 'wb') as out_file:
            pickle.dump(self.classifier, out_file)

    def read(self):
        with open(self.get_save_filename(), 'rb') as in_file:
            self.classifier = pickle.load(in_file)

