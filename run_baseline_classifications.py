# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle as pickle
import numpy as np
import os
import shutil
import sys

from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from utils import ne_uri_label_replace


MODELS = [
    ("LR", LogisticRegression),
    ("SVM", LinearSVC),
    ("MNB", MultinomialNB),
    ("RF", RandomForestClassifier),
    ("DT", DecisionTreeClassifier)
]


def run_classifier(model_name, model_class, features_type, dataset, labels, classes, indices):
    configs = {}

    if model_name in {"LR", "SVM", "RF"}:
        configs["verbose"] = 1

    if model_name in {"LR", "RF"}:
        configs["n_jobs"] = -1

    model = model_class(**configs)

    print('Fitting model', file=sys.stderr)
    model.fit(dataset[indices['train_indices']], integer_labels[indices['train_indices']])

    print('Classifying test set', file=sys.stderr)
    y_true = labels[indices['test_indices']]
    y_pred = model.predict(dataset[indices['test_indices']])

    print('Saving classification results', file=sys.stderr)
    save_dir = os.path.join(args.results_dir, model_name, features_type)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    header = ','.join(classes).encode('utf-8')

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    np.savetxt(os.path.join(save_dir, 'test_accuracy_NEU.txt'), np.array([accuracy], dtype=np.float32),
               fmt='%.3f'.encode('utf-8'), delimiter=','.encode('utf-8'))

    # Precision
    precision = precision_score(y_true, y_pred, average=None, labels=np.arange(classes.shape[0]))
    np.savetxt(os.path.join(save_dir, 'test_precision_NEU.txt'), precision.astype(np.float32),
               fmt='%.3f'.encode('utf-8'), delimiter=','.encode('utf-8'), header=header)

    # Recall
    recall = recall_score(y_true, y_pred, average=None, labels=np.arange(classes.shape[0]))
    np.savetxt(os.path.join(save_dir, 'test_recall_NEU.txt'), recall.astype(np.float32),
               fmt='%.3f'.encode('utf-8'), delimiter=','.encode('utf-8'), header=header)

    print('Finished handcrafted experiment for classifier {}'.format(model_name), file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=unicode)
    parser.add_argument("wvdataset", type=unicode)
    parser.add_argument("labels", type=unicode)
    parser.add_argument("wvlabels", type=unicode)
    parser.add_argument("indices", type=unicode)
    parser.add_argument("results_dir", type=unicode)
    parser.add_argument("--experiment_kind", type=unicode, default='legal')

    args = parser.parse_args()

    print('Loading dataset from file {}'.format(args.dataset), file=sys.stderr)

    # First run on handcrafted dataset
    dataset = np.load(args.dataset)
    dataset = csr_matrix((dataset['data'], dataset['indices'], dataset['indptr']), shape=dataset['shape'])

    print('Loading labels from file {}'.format(args.labels), file=sys.stderr)
    with open(args.labels, 'rb') as f:
        labels = pickle.load(f)

    print('Replacing the labels', file=sys.stderr)
    labels = list(ne_uri_label_replace(labels, None))

    print('Loading indices for train, test and validation', file=sys.stderr)
    indices = np.load(args.indices)

    print('Filtering dataset and labels according to indices', file=sys.stderr)
    dataset = dataset[indices['filtered_indices']]
    labels = np.array(labels)[indices['filtered_indices']]
    classes, integer_labels = np.unique(labels, return_inverse=True)

    print('Normalizing dataset', file=sys.stderr)
    dataset = normalize(dataset.astype(np.float32), norm='max', axis=0)

    for model_name, model_class in MODELS:
        print('Running handcrafted features dataset with {} classifier'.format(model_name), file=sys.stderr)

        try:
            run_classifier(model_name, model_class, 'handcrafted', dataset, labels, classes, indices)
        except Exception as e:
            print('The classifier {} throw an exception with message {}'.format(model_name, e.message), file=sys.stderr)

        print('Finished handcrafted experiments with {} classifier'.format(model_name), file=sys.stderr)

    print('Finished all handcrafted experiments', file=sys.stderr)

    print('Loading dataset from file {}. Filtering dataset according to indices'.format(args.wvdataset),
          file=sys.stderr)

    dataset = np.load(args.wvdataset)['dataset'][indices['filtered_indices']]

    print('Loading word vectors labels from file {}'.format(args.wvlabels), file=sys.stderr)
    with open(args.wvlabels, 'rb') as f:
        labels = pickle.load(f)

    print('Replacing the labels', file=sys.stderr)
    labels = list(ne_uri_label_replace(labels, None))

    print('Loading indices for train, test and validation', file=sys.stderr)
    indices = np.load(args.indices)

    print('Filtering labels according to indices', file=sys.stderr)
    labels = np.array(labels)[indices['filtered_indices']]
    classes, integer_labels = np.unique(labels, return_inverse=True)

    for model_name, model_class in MODELS:
        if model_name == "MNB":
            continue

        print('Running word vectors dataset with {} classifier'.format(model_name), file=sys.stderr)

        try:
            run_classifier(model_name, model_class, 'wordvectors', dataset, labels, classes, indices)
        except Exception as e:
            print('The classifier {} throw an exception with message {}'.format(model_name, e.message), file=sys.stderr)

        print('Finished word vectors experiments with {} classifier'.format(model_name), file=sys.stderr)

    print('Finished all experiments')
