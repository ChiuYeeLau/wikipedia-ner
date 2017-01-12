# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals
import pandas
import numpy

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)


class BaseClassifier(object):
    def __init__(self):
        self.test_results = pandas.DataFrame(
            columns=['accuracy', 'class', 'precision', 'recall', 'fscore'])
        if not hasattr(self, 'cl_iteration'):
            self.cl_iteration = 1

    def add_test_results(self, accuracy, precision, recall, fscore, classes,
                         y_true=None):
        self.test_results = self.test_results.append({'accuracy': accuracy},
                                                     ignore_index=True)
        if y_true is None:
            y_true = numpy.array([])
        for cls_idx, cls in enumerate(classes):
            self.test_results = self.test_results.append({
                'class': cls,
                'precision': precision[cls_idx],
                'recall': recall[cls_idx],
                'fscore': fscore[cls_idx],
                'support': (y_true == cls_idx).sum()
            }, ignore_index=True)

    def get_metrics(self, y_true, y_pred, return_extras=True):
        accuracy = accuracy_score(y_true, y_pred.astype(y_true.dtype))

        if not return_extras:
            return accuracy
        else:
            labels = numpy.arange(self.dataset.output_size(self.cl_iteration))
            precision = precision_score(y_true, y_pred, labels=labels,
                                        average=None)
            recall = recall_score(y_true, y_pred, labels=labels, average=None)
            fscore = f1_score(y_true, y_pred, labels=labels, average=None)

            return accuracy, precision, recall, fscore, y_true, y_pred

    def evaluate(self, dataset_name, *args, **kwargs):
        """Returns accuracy, precision, recall, fscore, y_true, y_pred"""
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError
