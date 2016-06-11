# -*- coding: utf-8 -*-

from __future__ import absolute_import

import numpy as np


class BaseClassifier(object):
    def __init__(self, dataset, labels, train_indices, test_indices, validation_indices):
        self.classes, integer_labels = np.unique(labels, return_inverse=True)

        self.input_size = dataset.shape[1]
        self.output_size = self.classes.shape[0]

        self.train_dataset = dataset[train_indices]
        self.train_labels = integer_labels[train_indices]
        self.test_dataset = dataset[test_indices]
        self.test_labels = integer_labels[test_indices]
        self.validation_dataset = dataset[validation_indices]
        self.validation_labels = integer_labels[validation_indices]
