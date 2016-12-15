# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import numpy as np
import sys
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=unicode)
    parser.add_argument("output", type=unicode)

    args = parser.parse_args()

    print('Loading dataset from file {}'.format(args.dataset), file=sys.stderr, flush=True)
    dataset = np.load(args.dataset)["dataset"]

    print('Saving dataset to file {}'.format(args.output), file=sys.stderr, flush=True)

    nnz_count = 0

    with open(args.output, 'w') as f:
        print('%%MatrixMarket matrix coordinate real general', file=f)
        print('%', file=f)
        print('{} {}'.format(dataset.shape[0], dataset.shape[1]), file=f)

        for i in tqdm(np.arange(dataset.shape[0])):
            for j in np.arange(dataset.shape[1]):
                if dataset[i, j] != 0:
                    nnz_count += 1
                    print('{} {} {}'.format(i+1, j+1, dataset[i, j]), file=f)

    print('Finished writing matrix.', file=sys.stderr, flush=True)
    print('Final number of non zero items: {}'.format(nnz_count), file=sys.stderr, flush=True)
    print('Sparse ratio: {:.02e}'.format(float(nnz_count) / float(dataset.shape[0] * dataset.shape[1])),
          file=sys.stderr, flush=True)
