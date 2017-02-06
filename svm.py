#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import sys

from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from wikipedianer.dataset import HandcraftedFeaturesDataset
from wikipedianer.pipeline.util import CL_ITERATIONS


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path',
                        type=str)
    parser.add_argument('labels_path',
                        type=str)
    parser.add_argument('indices_path',
                        type=str)
    parser.add_argument('results_path',
                        type=str)

    args = parser.parse_args()

    dataset = HandcraftedFeaturesDataset(args.dataset_path, args.labels_path, args.indices_path)

    for iidx, iteration in enumerate(CL_ITERATIONS[:-1]):
        print('Running for iteration %s' % iteration, file=sys.stderr)
        model = SGDClassifier(verbose=1, n_jobs=12)

        model.fit(dataset.train_dataset, dataset.train_labels[:, iidx])

        print('Getting results', file=sys.stderr)
        y_true = dataset.test_labels[:, iidx]
        y_pred = model.predict(dataset.test_dataset)
        results = pd.DataFrame(np.vstack([y_true, y_pred]).T, columns=['true', 'prediction'])

        print('Saving results', file=sys.stderr)
        results.to_csv(os.path.join(args.results_path, 'test_predictions_SVM_%s.csv' % iteration), index=False)
        
        print('Saving model', file=sys.stderr)
        joblib.dump(model, os.path.join(args.results_path, 'model_SVM_%s.pkl' % iteration))

    print('Finished all iterations', file=sys.stderr)

