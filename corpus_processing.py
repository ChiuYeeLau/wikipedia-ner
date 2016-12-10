# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys

from scipy.sparse import csr_matrix
from wikipedianer.pipeline.processing import (collect_gazeteers_and_subsample_non_entities, feature_selection,
                                              parse_corpus_to_handcrafted_features, parse_corpus_to_word_windows,
                                              parse_corpus_to_word_vectors, split_dataset, subsample_non_entities)

if sys.version_info.major == 3:
    unicode = str


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_path',
                        type=unicode,
                        help='Path to the corpus files (should be in column format).')
    parser.add_argument('--valid_indices_path',
                        type=unicode,
                        default=None,
                        help='Path to the valid indices obtained by subsampling the non entities. ' + \
                             'If it\'s not an existing file it will be created.')
    parser.add_argument('--handcrafted_matrix_path',
                        type=unicode,
                        default=None,
                        help='Path to store (or load if the file exists) the handcrafted features matrix.')
    parser.add_argument('--word_vectors_matrix_path',
                        type=unicode,
                        default=None,
                        help='Path to store the word vectors matrix.')
    parser.add_argument('--word_window_file_path',
                        type=unicode,
                        default=None,
                        help='Path to store the list of word tokens (to query the word2vec model).')
    parser.add_argument('--labels_file_path',
                        type=unicode,
                        default=None,
                        help='Path to store (or load if the file exists) the labels.')
    parser.add_argument('--word_vectors_model_file',
                        type=unicode,
                        default=None,
                        help='Path to the Word2Vec model file.')
    parser.add_argument('--features_file_path',
                        type=unicode,
                        default=None,
                        help='Path to store (or load if the file exists) the features names.')
    parser.add_argument('--filtered_handcrafted_matrix_path',
                        type=unicode,
                        default=None,
                        help='Path to store the handcrafted features matrix with feature selection.')
    parser.add_argument('--filtered_features_file_path',
                        type=unicode,
                        default=None,
                        help='Path to store the filtered features names.')
    parser.add_argument('--gazetteer_file_path',
                        type=unicode,
                        default=None,
                        help='Path to store (or load if it exists) the gazetteer file.')
    parser.add_argument('--indices_save_path',
                        type=unicode,
                        default=None,
                        help='Path to save the indices file.')
    parser.add_argument('--classes_save_path',
                        type=unicode,
                        default=None,
                        help='Path to save the classes file.')
    parser.add_argument('--max_features',
                        type=int,
                        default=12000,
                        help='Number maximum of features to take in the handcrafted feature selection.')
    parser.add_argument('--word_vectors_window',
                        type=int,
                        default=3,
                        help='Word vectors window (symmetrical window size).')
    parser.add_argument('--train_size',
                        type=float,
                        default=0.8,
                        help='Size of the train set.')
    parser.add_argument('--test_size',
                        type=float,
                        default=0.1,
                        help='Size of the test set.')
    parser.add_argument('--validation_size',
                        type=float,
                        default=0.1,
                        help='Size of the validation set.')
    parser.add_argument('--min_count',
                        type=int,
                        default=10,
                        help='Minimum amount of occurrences of a class in the corpus.')
    parser.add_argument('--remove_stopwords',
                        action='store_true',
                        help='Whether to include or not stopwords (outside entities).')
    parser.add_argument('--debug_word_vectors',
                        action='store_true',
                        help='Helps debugging word vectors operations.')

    args = parser.parse_args()

    if not os.path.isdir(args.corpus_path):
        print('The given corpus path (%s) is not a valid directory' % args.corpus_path, file=sys.stderr)
        sys.exit(os.EX_USAGE)

    labels = None
    if args.labels_file_path is None:
        print('You must provide a path for the labels file', file=sys.stderr)
        sys.exit(os.EX_USAGE)
    elif os.path.isfile(args.labels_file_path):
        with open(args.labels_file_path, 'rb') as f:
            labels = pickle.load(f)

    if args.valid_indices_path is None:
        print('You must provide a path for the valid indices file.', file=sys.stderr)
        sys.exit(os.EX_USAGE)

    print('Getting valid indices', file=sys.stderr)
    if os.path.isfile(args.valid_indices_path):
        with open(args.valid_indices_path, 'rb') as f:
            valid_indices = pickle.load(f)
    elif args.gazetteer_file_path is not None and os.path.isfile(args.gazetteer_file_path):
        valid_indices = subsample_non_entities(args.corpus_path, args.valid_indices_path,
                                               remove_stopwords=args.remove_stopwords)

    if args.handcrafted_matrix_path is not None:
        if args.gazetteer_file_path is None:
            print('You must provide a path for the gazetteer file', file=sys.stderr)
            sys.exit(os.EX_USAGE)
        if args.features_file_path is None:
            print('You must provide a path for the features file', file=sys.stderr)
            sys.exit(os.EX_USAGE)
        if args.filtered_features_file_path is None:
            print('You must provide a path for the filtered features file', file=sys.stderr)
            sys.exit(os.EX_USAGE)
        if args.filtered_handcrafted_matrix_path is None:
            print('You must provide a path for the handcrafted features matrix with feature selection file',
                  file=sys.stderr)
            sys.exit(os.EX_USAGE)

        print('Getting gazetteer', file=sys.stderr)
        if os.path.isfile(args.gazetteer_file_path):
            with open(args.gazetteer_file_path, 'rb') as f:
                gazetteer, sloppy_gazetteer = pickle.load(f)
        else:
            gazetteer, sloppy_gazetteer, valid_indices = \
                collect_gazeteers_and_subsample_non_entities(args.corpus_path, args.gazetteer_file_path,
                                                             args.valid_indices_path, args.remove_stopwords)

        print('Getting handcrafted features matrix, labels and features names', file=sys.stderr)
        if os.path.isfile(args.handcrafted_matrix_path) and os.path.isfile(args.labels_file_path) and \
                os.path.isfile(args.features_file_path):
            dataset = np.load(args.handcrafted_matrix_path)
            dataset = csr_matrix((dataset['data'], dataset['indices'], dataset['indptr']), shape=dataset['shape'])

            with open(args.labels_file_path, 'rb') as f:
                labels = pickle.load(f)

            with open(args.features_file_path, 'rb') as f:
                feature_names = pickle.load(f)
        else:
            dataset, labels, feature_names = \
                parse_corpus_to_handcrafted_features(args.corpus_path, args.handcrafted_matrix_path,
                                                     args.labels_file_path, args.features_file_path,
                                                     remove_stopwords=args.remove_stopwords,
                                                     gazetteer=gazetteer, sloppy_gazetteer=sloppy_gazetteer,
                                                     valid_indices=valid_indices)

        # Free a little memory
        del gazetteer, sloppy_gazetteer

        feature_selection(dataset, feature_names, args.filtered_handcrafted_matrix_path,
                          args.filtered_features_file_path, args.max_features)

        # Free a little more memory
        del dataset, feature_names

    if args.word_vectors_matrix_path is not None:
        labels_file_path = args.labels_file_path if labels is None else None

        print('Getting word vectors matrix and labels', file=sys.stderr)
        word_vectors_labels = \
            parse_corpus_to_word_vectors(args.corpus_path, args.word_vectors_matrix_path, args.word_vectors_model_file,
                                         labels_file_path=labels_file_path, remove_stopwords=args.remove_stopwords,
                                         valid_indices=valid_indices, window=args.word_vectors_window,
                                         debug=args.debug_word_vectors)

        labels = word_vectors_labels if labels is None else labels

    if args.word_window_file_path is not None:
        labels_file_path = args.labels_file_path if labels is None else None

        print('Getting word windows and labels', file=sys.stderr)
        word_vectors_labels = \
            parse_corpus_to_word_windows(args.corpus_path, args.word_window_file_path, labels_file_path,
                                         remove_stopwords=args.remove_stopwords, valid_indices=valid_indices,
                                         window=args.word_vectors_window)

        labels = word_vectors_labels if labels is None else labels

    if args.indices_save_path is not None and args.classes_save_path is not None:
        split_dataset(labels, args.indices_save_path, args.classes_save_path, train_size=args.train_size,
                      test_size=args.test_size, validation_size=args.validation_size, min_count=args.min_count)

