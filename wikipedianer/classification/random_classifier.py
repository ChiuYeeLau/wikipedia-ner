"""Classifier that predicts random weighted classes."""

from __future__ import absolute_import, print_function, unicode_literals

from sklearn import metrics

import numpy
import pandas
import os

from .base import BaseClassifier
from .double_step_classifier import ClassifierFactory


class RandomClassifierFactory(ClassifierFactory):
    def get_classifier(self, dataset, *args, **kwargs):
        return RandomClassifier(dataset)


class RandomClassifier(BaseClassifier):

    def __init__(self, dataset, cl_iteration=1):
        super(RandomClassifier, self).__init__()
        self.dataset = dataset
        self.cl_iteration = cl_iteration
        self.unique_elements = None
        self.probabilities = None

    def train(self, *args, **kwargs):
        # Count the unique elements
        elements, counts = numpy.unique(
            self.dataset.datasets['train'].labels[:,self.cl_iteration],
            return_counts=True)
        self.unique_elements = elements
        self.probabilities = (counts / counts.sum()).astype(numpy.float64)

        # Get accuracy on test dataset
        accuracy, precision, recall, fscore, y_true, y_pred = self.evaluate(
            'test', return_extras=True)
        self.add_test_results(accuracy, precision, recall, fscore,
                              self.dataset.classes[1])

    def evaluate(self, dataset_name='test', return_extras=False, *args,
                 **kwargs):
        y_true = self.dataset.datasets[dataset_name].labels[:,self.cl_iteration]
        if self.unique_elements is None or self.probabilities is None:
            self.train()
        y_pred = numpy.random.choice(
            self.unique_elements, size=self.dataset.num_examples(dataset_name),
            p=self.probabilities)

        accuracy = metrics.accuracy_score(y_true, y_pred.astype(y_true.dtype))
        if not return_extras:
            return accuracy
        else:
            labels = numpy.arange(
                self.dataset.output_size(self.cl_iteration))
            precision = metrics.precision_score(y_true, y_pred, labels=labels,
                                                average=None)
            recall = metrics.recall_score(y_true, y_pred, labels=labels,
                                          average=None)
            fscore = metrics.f1_score(y_true, y_pred, labels=labels,
                                      average=None)

            return accuracy, precision, recall, fscore, y_true, y_pred
