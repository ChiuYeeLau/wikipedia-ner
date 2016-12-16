# -*- coding: utf-8 -*-
"""Script to train a classifier for NE linking using a double step classifier.
"""

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import logging

logging.basicConfig(level=logging.INFO)
import numpy
import pickle

from scipy.sparse import csr_matrix
from .wikipedianer.classification import double_step_classifier


def read_arguments():
    """Parses the arguments from the stdin and returns an object."""
    parser = argparse.ArgumentParser()
    parser.add_argument('input_matrix', type=str,
                        help='Path of file with the numpy matrix used for'
                             'training')
    parser.add_argument('--high_level_labels', '-hl,', type=str,
                        help='Path of file with the pickled labels to use as'
                             'first classification step.')
    parser.add_argument('--low_level_labels', '-ll', type=str,
                        help='Path of file with the pickled labels to use as'
                             'first classification step.')
    parser.add_argument('--high_level_model', '-m', type=str,
                        help='Path of file with the tensorflow model trained'
                             'with high level labels')
    parser.add_argument('--results_dirname', '-r', type=str,
                        help='Path of directory to save results')
    parser.add_argument('--models_dirname', '-m', type=str,
                        help='Path of directory to save the trained models.')
    parser.add_argument('--indices', '-i', type=float, nargs=3,
                        help='Path of file with the numpy matrix containing'
                             'the split indices training/testing/validation')


    return parser.parse_args()


def read_dataset(dataset_filename, high_labels_filename, low_labels_filename):
    logging.info('Loading dataset from file {}'.format(dataset_filename))

    dataset = numpy.load(dataset_filename)
    dataset = csr_matrix(
        (dataset['data'], dataset['indices'], dataset['indptr']),
        shape=dataset['shape'])

    logging.info('Loading labels from file {}'.format(high_labels_filename))
    # Labels is a list of strings.
    with open(high_labels_filename, 'rb') as f:
        hl_labels = pickle.load(f)

    logging.info('Loading labels from file {}'.format(low_labels_filename))
    # Labels is a list of strings.
    with open(high_labels_filename, 'rb') as f:
        ll_labels = pickle.load(f)
    return dataset, numpy.array(hl_labels), numpy.array(ll_labels)


def main():
    """Main function of script."""
    args = read_arguments()
    dataset, hl_labels, ll_labels = read_dataset(args.input_matrix, args.labels)
    indices = numpy.load(args.indices)
    classifier = double_step_classifier.DoubleStepClassifier(
        dataset, hl_labels, ll_labels, indices['train_indices'],
        indices['test_indices'], indices['validation_indices'],
        args.models_dirname, args.results_dirname)

    classifier.train()
    classifier.evaluate()



if __name__ == '__main__':
    main()