# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import os
import sys

from wikipedianer.pipeline.classification import get_dataset
from wikipedianer.pipeline.util import CL_ITERATIONS
from wikipedianer.classification.nn_classifier import \
    NNeighborsClassifierFactory


def run_knn_classifier(dataset, results_save_path, cl_iterations,
                       features_filepath):

    factory = NNeighborsClassifierFactory(features_filepath, results_save_path)

    for iteration in cl_iterations:
        experiment_name = CL_ITERATIONS[iteration]
        print('Running experiment: %s' % iteration, file=sys.stderr, flush=True)
        knn = factory.get_classifier(dataset, experiment_name,
                                     cl_iteration=iteration)
        print('Training the classifier', file=sys.stderr, flush=True)
        knn.train()

        print('Finished experiment %s' % experiment_name, file=sys.stderr, flush=True)

    print('Finished all the experiments', file=sys.stderr, flush=True)


def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path',
                        type=str,
                        help='Path to the dataset file.')
    parser.add_argument('labels_path',
                        type=str,
                        help='Path to the labels file.')
    parser.add_argument('indices_path',
                        type=str,
                        help='Path to the indices file.')
    parser.add_argument('results_save_path',
                        type=str,
                        help='Path to the directory to store the results.')
    parser.add_argument('--word_vectors_path',
                        type=str,
                        default=None,
                        help='Path to word vectors models file. ' +
                             'If given it assumes the classification uses word vectors.')
    parser.add_argument('--features_filepath',
                        type=str,
                        default=None,
                        help='Path to pickled file with feature names. ')
    parser.add_argument('--cl_iterations',
                        type=str,
                        nargs='+',
                        default=[],
                        help='Strings to determine the iterations in CL to run.'
                             'The classifier is trained always in batch')
    parser.add_argument('--classes_path', type=str, default=None,
                        help='File with the classes to replace labels')

    return parser.parse_args()

def main():
    args = read_arguments()
    if not args.cl_iterations:
        print('You have to provide a valid set of iterations for CL', file=sys.stderr, flush=True)
        sys.exit(os.EX_USAGE)

    args.cl_iterations = [args.cl_iterations] if isinstance(args.cl_iterations, str) else args.cl_iterations
    cl_iterations = [CL_ITERATIONS.index(cl_iter) for cl_iter in args.cl_iterations]

    dataset = get_dataset(
        dataset_path=args.dataset_path, labels_path=args.labels_path,
        indices_path=args.indices_path, classes_path=args.classes_path,
        word_vectors_path=args.word_vectors_path)

    run_knn_classifier(
        dataset, results_save_path=args.results_save_path,
        cl_iterations=cl_iterations, features_filepath=args.features_filepath)


if __name__ == '__main__':
    main()