# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import numpy as np
import sys


class BaseClassifier(object):
    def __init__(self, dataset, labels, train_indices, test_indices, validation_indices, filtered_indices=None):
        print('Getting unique classes', file=sys.stderr)
        self.classes, integer_labels = np.unique(labels, return_inverse=True)

        self.input_size = dataset.shape[1]
        self.output_size = self.classes.shape[0]

        print('Indexing split datasets', file=sys.stderr)
        filtered_indices = np.arange(dataset.shape[0]) if filtered_indices is None else filtered_indices
        self.train_dataset = dataset[filtered_indices[train_indices]]
        self.train_labels = integer_labels[train_indices].astype(np.int32)
        self.test_dataset = dataset[filtered_indices[test_indices]]
        self.test_labels = integer_labels[test_indices].astype(np.int32)
        self.validation_dataset = dataset[filtered_indices[validation_indices]]
        self.validation_labels = integer_labels[validation_indices].astype(np.int32)
