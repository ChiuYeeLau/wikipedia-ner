# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals
import pandas
import numpy

class BaseClassifier(object):
    def __init__(self):
        self.test_results = pandas.DataFrame(
            columns=['accuracy', 'class', 'precision', 'recall', 'fscore'])

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

    def evaluate(self, dataset_name, *args, **kwargs):
        raise NotImplementedError

    def train(self, *args, **kwargs):
        raise NotImplementedError
