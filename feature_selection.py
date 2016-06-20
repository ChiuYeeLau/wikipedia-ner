#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import numpy as np
import sys
from scipy.sparse import csr_matrix, csc_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=unicode)
    parser.add_argument("output_file", type=unicode)
    parser.add_argument("--max_features", type=int, default=10000)

    args = parser.parse_args()

    print('Loading dataset from file {}'.format(args.input_file), file=sys.stderr)
    dataset = np.load(args.input_file)
    dataset = csc_matrix(csr_matrix((dataset['data'], dataset['indices'], dataset['indptr']), shape=dataset['shape']))

    print('Calculating variance of dataset features', file=sys.stderr)
    square_dataset = dataset.copy()
    square_dataset.data **= 2
    variance = np.asarray(square_dataset.mean(axis=0) - np.square(dataset.mean(axis=0)))[0]

    print('Sorting indices of variance', file=sys.stderr)
    sorted_variance_indices = np.argsort(variance)[::-1]

    print('Selecting top variance features', file=sys.stderr)
    dataset = csr_matrix(dataset[:, sorted_variance_indices[:args.max_features]])

    print('Saving dataset to file {}'.format(args.output_file), file=sys.stderr)
    np.savez_compressed(args.output_file, data=dataset.data, indices=dataset.indices,
                        indptr=dataset.indptr, shape=dataset.shape)

    print('All operations finished')
