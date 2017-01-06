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

    def predict(self, x_matrix):
        return self.classifier.predict(x_matrix[:, self.valid_features])

    def evaluate(self, dataset_name='test', restore=False, *args, **kwargs):
        if restore and not self.trained:
            self.read()
        logging.info('Obtaining predictions')
        y_pred = self.predict(self.dataset.datasets[dataset_name].data)
        y_true = self.dataset.dataset_labels(dataset_name, 1)
        predictions = pandas.DataFrame(numpy.vstack([y_true, y_pred]).T,
                                       columns=['true', 'prediction'])
        predictions.to_csv(self.get_predictions_filename(), index=False)
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

    def get_save_filename(self):
        return os.path.join(self.results_save_path,
                            '{}_knn.model'.format(self.experiment_name))

    def get_predictions_filename(self):
        return os.path.join(
            self.results_save_path,
            'test_predictions_{}.csv'.format(self.experiment_name))

    def get_results_filename(self):
        return os.path.join(
            self.results_save_path,
            'test_results_{}.csv'.format(self.experiment_name))

    def save(self):
        with open(self.get_save_filename(), 'wb') as out_file:
            pickle.dump(self.classifier, out_file)

    def read(self):
        with open(self.get_save_filename(), 'rb') as in_file:
            self.classifier = pickle.load(in_file)

