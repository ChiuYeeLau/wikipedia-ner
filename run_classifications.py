# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import numpy as np
import os
import sys

from wikipedianer.pipeline.classification import run_classifier
from wikipedianer.pipeline.util import CL_ITERATIONS

# To avoid error induced by chance
np.random.seed(0)


if __name__ == '__main__':
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
    parser.add_argument('--weights_save_path',
                        type=str,
                        default='',
                        help='Path to save the pre-trained weights.')
    parser.add_argument('--layers',
                        type=int,
                        nargs='+',
                        default=[],
                        help='Layers of the network.')
    parser.add_argument('--cl_iterations',
                        type=str,
                        nargs='+',
                        default=[],
                        help='Strings to determine the iterations in CL.')
    parser.add_argument('--save_models',
                        type=str,
                        default=[],
                        nargs='+',
                        help='String to determine the models of the experiment to save.')
    parser.add_argument('--completed_iterations',
                        type=str,
                        default=[],
                        nargs='+',
                        help='Iterations completed and ready to do a follow up')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01)
    parser.add_argument('--epochs',
                        type=int,
                        default=10000)
    parser.add_argument('--batch_size',
                        type=int,
                        default=2100)
    parser.add_argument('--loss_report',
                        type=int,
                        default=250)
    parser.add_argument('--dropout_ratios',
                        type=float,
                        default=[],
                        nargs='+')
    parser.add_argument('--batch_normalization',
                        action='store_true')
    parser.add_argument('--debug_word_vectors',
                        action='store_true')

    args = parser.parse_args()

    if not args.layers:
        print('You have to provide a valid architecture.', file=sys.stderr, flush=True)
        sys.exit(os.EX_USAGE)

    args.layers = [args.layers] if isinstance(args.layers, int) else args.layers

    if not args.cl_iterations:
        print('You have to provide a valid set of iterations for CL', file=sys.stderr, flush=True)
        sys.exit(os.EX_USAGE)

    args.cl_iterations = [args.cl_iterations] if isinstance(args.cl_iterations, str) else args.cl_iterations
    cl_iterations = [CL_ITERATIONS.index(cl_iter) for cl_iter in args.cl_iterations]

    save_models = [cl_iter in set(args.save_models) for cl_iter in CL_ITERATIONS]

    args.dropout_ratios = [args.dropout_ratios] if isinstance(args.dropout_ratios, int) else args.dropout_ratios

    args.completed_iterations = [args.completed_iterations] if isinstance(args.completed_iterations, str) \
        else args.completed_iterations
    completed_iterations = [CL_ITERATIONS.index(cl_iter) for cl_iter in args.completed_iterations]

    run_classifier(dataset_path=args.dataset_path, labels_path=args.labels_path, indices_path=args.indices_path,
                   results_save_path=args.results_save_path, pre_trained_weights_save_path=args.weights_save_path,
                   cl_iterations=cl_iterations, word_vectors_path=args.word_vectors_path, layers=args.layers,
                   dropout_ratios=args.dropout_ratios, save_models=save_models,
                   completed_iterations=completed_iterations, learning_rate=args.learning_rate, epochs=args.epochs,
                   batch_size=args.batch_size, loss_report=args.loss_report,
                   batch_normalization=args.batch_normalization, debug_word_vectors=args.debug_word_vectors)
