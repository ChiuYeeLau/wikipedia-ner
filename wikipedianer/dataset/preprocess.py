# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import numpy as np


class StratifiedSplitter(object):
    def __init__(self, labels, filtered_indices):
        self._classes, self._y_indices = np.unique(labels[filtered_indices], return_inverse=True)
        self._filtered_indices = filtered_indices
        self._train_indices = np.array([])
        self._test_indices = np.array([])
        self._validation_indices = np.array([])

    def split_dataset(self, train_size=0.8, test_size=0.1, validation_size=0.1):
        assert train_size + test_size + validation_size == 1.

        y_counts = np.bincount(self._y_indices)
        n_cls = self._classes.shape[0]
        n_train = self._y_indices.shape[0] * train_size
        n_test = self._y_indices.shape[0] * test_size
        n_validation = self._y_indices.shape[0] * validation_size

        if (validation_size > 0. and np.min(y_counts) < 3) or (validation_size == 0. and np.min(y_counts) < 2):
            raise ValueError("The least populated class needs to have more than 2 occurrences" +
                             " if you want to split with validation or more than 1 occurrence otherwise.")

        if n_train < n_cls:
            raise ValueError('The train_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (n_train, n_cls))
        if n_test < n_cls:
            raise ValueError('The test_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (n_test, n_cls))
        if validation_size > 0 and n_validation < n_cls:
            raise ValueError('The validation_size = %d should be greater or '
                             'equal to the number of classes = %d' %
                             (n_validation, n_cls))

        test_count = np.maximum(np.round(y_counts * test_size), np.ones(n_cls)).astype(np.int32)
        train_count = (y_counts - test_count).astype(np.int32)
        if validation_size > 0:
            validation_count = np.maximum(np.round(train_count * validation_size), np.ones(n_cls)).astype(np.int32)
            train_count -= validation_count

        train_indices = []
        test_indices = []
        validation_indices = []

        for idx, cls in enumerate(self._classes):
            perm_cls = np.random.permutation(np.where(self._y_indices == idx)[0])

            train_indices.extend(perm_cls[:train_count[idx]])
            test_indices.extend(perm_cls[train_count[idx]:train_count[idx]+test_count[idx]])

            if validation_size > 0:
                validation_indices.extend(perm_cls[train_count[idx]+test_count[idx]:])

        self._train_indices = np.random.permutation(np.array(train_indices, dtype=np.int32))
        self._test_indices = np.random.permutation(np.array(test_indices, dtype=np.int32))
        self._validation_indices = np.random.permutation(np.array(validation_indices, dtype=np.int32))

    def save_splitted_dataset_indices(self, path, train_size=0.8, test_size=0.2, validation_size=0.2):
        if self._train_indices.shape == 0:
            self.split_dataset(train_size, test_size, validation_size)

        np.savez_compressed(path, train_indices=self._train_indices, test_indices=self._test_indices,
                            validation_indices=self._validation_indices, filtered_indices=self._filtered_indices)
