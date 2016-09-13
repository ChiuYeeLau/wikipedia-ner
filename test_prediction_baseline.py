# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import numpy as np
import sys
from scipy.sparse import csr_matrix
from sklearn.externals import joblib
from sklearn.preprocessing import normalize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=unicode)
    parser.add_argument('indices', type=unicode)
    parser.add_argument('model', type=unicode)
    parser.add_argument('results', type=unicode)
    parser.add_argument('--word_vectors', action='store_true')

    args = parser.parse_args()

    print('Loading dataset from file {}'.format(args.dataset), file=sys.stderr)
    if args.word_vectors:
        dataset = np.load(args.dataset)['dataset']
    else:
        dataset = np.load(args.dataset)
        dataset = csr_matrix((dataset['data'], dataset['indices'], dataset['indptr']), shape=dataset['shape'])

        print('Normalizing dataset', file=sys.stderr)
        dataset = normalize(dataset.astype(np.float32), norm='max', axis=0)

    print('Loading indices from file {}'.format(args.indices), file=sys.stderr)
    indices = np.load(args.indices)

    print('Getting test dataset', file=sys.stderr)
    dataset = dataset[indices['filtered_indices']]
    dataset = dataset[indices['test_indices']]

    model = joblib.load(args.model)

    y_pred = model.predict(dataset)

    print('Saving results to file {}'.format(args.results), file=sys.stderr)
    np.savetxt(args.results, y_pred, fmt='%d'.encode('utf-8'))

    print('All finished', file=sys.stderr)
