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
    parser.add_argument("--min_variance", type=float, default=2e-4)

    args = parser.parse_args()

    print('Loading dataset from file {}'.format(args.input_file), file=sys.stderr)
    dataset = np.load(args.input_file)
    dataset = csc_matrix(csr_matrix((dataset['data'], dataset['indices'], dataset['indptr']), shape=dataset['shape']))

    print('Calculating variance of dataset features', file=sys.stderr)
    square_dataset = dataset.copy()
    square_dataset.data **= 2
    variance = np.asarray(square_dataset.mean(axis=0) - np.square(dataset.mean(axis=0)))[0]

    print('Getting features with variance over {:.2e}'.format(args.min_variance), file=sys.stderr)
    valid_indices = np.where(variance >= args.min_variance)[0]

    print('Filtering features', file=sys.stderr)
    dataset = csr_matrix(dataset[:, valid_indices])

    print('Final features count: {}'.format(dataset.shape[1]))

    print('Saving dataset to file {}'.format(args.output_file), file=sys.stderr)
    np.savez_compressed(args.output_file, data=dataset.data, indices=dataset.indices,
                        indptr=dataset.indptr, shape=dataset.shape)

    print('All operations finished')
