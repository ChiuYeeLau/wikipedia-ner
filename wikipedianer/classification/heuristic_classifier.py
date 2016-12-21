"""Heuristic classifier for Named Entity Linking using partial matches"""

import numpy
import pandas
import random

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class HeuristicClassifier(object):

    DEFAULT_LABEL = 'O'

    def __init__(self, dataset, entities, features):
        self.dataset = dataset
        self.entities = entities
        self.features_map = features
        # We ensure keys have always the same order
        self.features_keys = sorted(list(self.features_map.keys()))
        self.mask = numpy.zeros((len(self.features_keys),))
        self.default_index = None
        self._add_O_class()
        self.test_results = pandas.DataFrame(
            columns=['accuracy', 'class', 'precision', 'recall', 'fscore'])

    def _add_O_class(self):
        index = numpy.where(self.dataset.classes[1] ==
                                         self.DEFAULT_LABEL)[0]
        if index.shape[0] == 0:  # Default label not in classes
            self.default_index = self.dataset.classes[1].shape[0]
            self.dataset.classes = (
                self.dataset.classes[0],
                numpy.append(self.dataset.classes[1], self.DEFAULT_LABEL))
        else:
            self.default_index = index[0]

    def get_token(self, instance):
        feature_index = numpy.where(
            instance[self.mask, self.features_keys] == 1)[1]
        if feature_index.shape[0] == 0:
            return None
        return self.features_map[self.features_keys[feature_index]]

    def get_numeric_label(self, label):
        label_index = numpy.where(self.dataset.classes[1] == label)[0]
        if label_index.shape[0] == 0:
            return self.default_index
        return label_index[0]

    def predict(self, x_matrix):
        predictions = []
        for instance in x_matrix:
            token = self.get_token(instance)
            prediction = self.entities.get(token)  # if exists, is a set
            if prediction is None or len(prediction) == 0:
                prediction = 'O'
            else:
                prediction = 'I-' + random.sample(prediction, 1)[0]
            predictions.append(self.get_numeric_label(prediction))
        return numpy.array(predictions)

    def train(self, save_layers=False):
        # Get test accuracy
        y_pred = self.predict(self.dataset.datasets['test'].data)
        y_true = self.dataset.dataset_labels('test', 1)
        assert y_pred.shape[0] == y_true.shape[0]
        accuracy = accuracy_score(y_true, y_pred.astype(y_true.dtype))

        precision = precision_score(y_true, y_pred, labels=numpy.arange(
            self.dataset.output_size(1)), average=None)
        recall = recall_score(y_true, y_pred, labels=numpy.arange(
            self.dataset.output_size(1)), average=None)
        fscore = f1_score(y_true, y_pred, labels=numpy.arange(
            self.dataset.output_size(1)), average=None)

        self.test_results = self.test_results.append({'accuracy': accuracy},
                                                     ignore_index=True)
        for cls_idx, cls in enumerate(self.dataset.classes[1]):
            self.test_results = self.test_results.append({
                'class': cls,
                'precision': precision[cls_idx],
                'recall': recall[cls_idx],
                'fscore': fscore[cls_idx]
            }, ignore_index=True)

