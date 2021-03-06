"""Heuristic classifier for Named Entity Linking using partial matches"""
import logging
import numpy
import pickle
import random

from collections import defaultdict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)
from .base import BaseClassifier
from .double_step_classifier import ClassifierFactory


class HeuristicClassifierFactory(ClassifierFactory):
    def __init__(self, features_filename):
        # Read the features
        with open(features_filename, 'rb') as feature_file:
            features = pickle.load(feature_file)
            # Get indices for each current token
        self.token_features = [
            index for index, x in enumerate(features)
            if x.startswith('token:current')]
        self.prev_features = [
            index for index, x in enumerate(features)
            if x.startswith('token:prev')]
        self.next_features = [
            index for index, x in enumerate(features)
            if x.startswith('token:next')]
        self.total_features = len(features)

    def get_classifier(self, dataset, experiment_name):
        assert self.total_features == dataset.datasets['train'].data.shape[1]
        return HeuristicClassifier(dataset, self.token_features,
                                   self.prev_features, self.next_features)

def get_ngram_set(token, prev, next_):
    return set([(token, ), (prev, token), (token, next_), (prev, token, next_)])

MAX_POSSIBLE_SCORE = len(get_ngram_set(None, None, None))


def predict_instance(classifier, instance):
    token_index = classifier.get_token_index(instance, classifier.token_features)
    possible_labels = classifier.token_to_label_map.get(token_index, [])
    if len(possible_labels) is 0:
        return random.randrange(
            0, classifier.dataset.classes[1].shape[0])
    prev_index = classifier.get_token_index(instance, classifier.prev_features)
    next_index = classifier.get_token_index(instance, classifier.next_features)
    max_score = 0
    selected_label = None
    for label in possible_labels:
        ngrams = get_ngram_set(token_index, prev_index, next_index)
        label_score = len(ngrams.intersection(classifier.n_gram_map.get(
            label, set())))
        if max_score < label_score:
            max_score = label_score
            selected_label = label
        if max_score == MAX_POSSIBLE_SCORE:
            break
    return selected_label


class HeuristicClassifier(BaseClassifier):

    DEFAULT_LABEL = 'O'

    def __init__(self, dataset, token_features, prev_features, next_features):
        """

        :param dataset:
        :param entities:
        :param features: A dictionary from indexes in the feature array to
         tokens. Only indices of features indicating the current token are used.
        """
        super(HeuristicClassifier, self).__init__()
        self.dataset = dataset
        # A map from each token index in the training dataset to the
        # list of labels it has been tagged with.
        self.token_to_label_map = defaultdict(set)
        # A map from each label to a set of unigrams, bigrams and trigrams
        # (ordered) extracted from the label mentions in the train dataset.
        self.n_gram_map = defaultdict(set)

        self.token_features = token_features
        self.next_features = next_features
        self.prev_features = prev_features

    def get_token_index(self, instance, possible_values):
        feature_index = instance[:, possible_values].nonzero()[1]
        if feature_index.shape[0] == 0:
            return None
        return feature_index[0]

    def predict(self, x_matrix):
        num_cores = multiprocessing.cpu_count()
        logging.info('Paralelizing over {} cores'.format(num_cores))
        predictions = Parallel(n_jobs=num_cores)(delayed(
            predict_instance)(self, instance) for instance in x_matrix)
        return numpy.array(predictions).astype(numpy.int32)

    def evaluate(self, *args, **kwargs):
        if len(self.token_to_label_map) == 0:
            self.train()
        logging.info('Obtaining predictions')
        y_pred = self.predict(self.dataset.datasets['test'].data)
        y_true = self.dataset.dataset_labels('test', 1)
        assert y_pred.shape[0] == y_true.shape[0]
        accuracy = accuracy_score(y_true, y_pred.astype(y_true.dtype))

        labels = numpy.arange(self.dataset.output_size(1))

        precision = precision_score(y_true, y_pred, labels=labels, average=None)
        recall = recall_score(y_true, y_pred, labels=labels, average=None)
        fscore = f1_score(y_true, y_pred, labels=labels, average=None)
        return accuracy, precision, recall, fscore, y_true, y_pred

    def train(self, save_layers=False):
        # Count all instances
        no_token_instances = 0
        for index, instance in enumerate(self.dataset.datasets['train'].data):
            token_index = self.get_token_index(instance, self.token_features)
            if not token_index:
                no_token_instances += 1
                continue
            prev_index = self.get_token_index(instance, self.prev_features)
            next_index = self.get_token_index(instance, self.next_features)

            label = self.dataset.datasets['train'].labels[index][1]
            self.token_to_label_map[token_index].add(label)
            self.n_gram_map[label].update(get_ngram_set(
                token_index, prev_index, next_index))
        logging.info('Training completed.')
        logging.info('No token instances: {0:.2f}'.format(
            no_token_instances/float(self.dataset.num_examples('train'))))
        logging.info('Average labels per token: {}'.format(
            numpy.mean([len(x) for x in self.token_to_label_map.values()])))

        # accuracy, precision, recall, fscore, y_true, y_pred = self.evaluate()

        # self.add_test_results(accuracy, precision, recall, fscore,
        #                     classes=self.dataset.classes[1], y_true=y_true)

