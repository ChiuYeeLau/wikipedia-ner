# -*- coding: utf-8 -*-
"""Script to train a classifier for NE linking using a double step classifier.

python run_classifications_nel.py ../data/legal_sampled/filtered_handcreafter_matrix.npz  --labels_filepath ../data/legal_sampled/labels.pickle --results_dirname ../results/double_step_classifier/sample/ --indices ../data/legal_sampled/indices.npz --classifier 'heuristic' --features_filename ../data/legal_sampled/filtered_features_file_path --entities_filename ../data/mappings/all_uris_labels.pickle
"""

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import os

import jsonlines
import logging
import pickle

from collections import defaultdict

import numpy
import pandas

logging.basicConfig(level=logging.INFO)

from wikipedianer.classification import double_step_classifier
from wikipedianer.classification import heuristic_classifier
from wikipedianer.classification import logistic_regression
from wikipedianer.classification import random_classifier
from wikipedianer.classification import nn_classifier
from wikipedianer.dataset import HandcraftedFeaturesDataset



def read_arguments():
    """Parses the arguments from the stdin and returns an object."""
    parser = argparse.ArgumentParser()
    parser.add_argument('input_matrix_file', type=str,
                        help='Path of file with the numpy matrix used for'
                             'training')
    parser.add_argument('--labels_filepath', '-l,', type=str,
                        help='Path of file with the pickled labels to use')
    parser.add_argument('--high_level_model', type=str, default=None,
                        help='Path of file with the tensorflow model trained'
                             'with high level labels')
    parser.add_argument('--results_dirname', '-r', type=str,
                        help='Path of directory to save results')
    parser.add_argument('--indices', '-i', type=str,
                        help='Path of file with the numpy matrix containing'
                             'the split indices training/testing/validation')
    parser.add_argument('--classifier', type=str, default='mlp',
                        help='The classifier to use. Possible values are mlp '
                             'lr, random and heuristic.')
    parser.add_argument('--features_filename', type=str, default=None,
                        help='Path of file with the filtered features. Only '
                             'for heuristic classifier.')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Do not train the classifier, only perform'
                             'evaluation')
    parser.add_argument('--use-trained', action='store_true',
                        help='Check if classifier is already stored to avoid'
                             'training.')
    parser.add_argument('--dropout-ratio', type=float, default=0.0,
                        help='The dropout ratio for the classifier.')
    parser.add_argument('--num-layers', type=int, default=1,
                        help='Use a hidden layer in the mlp classifier.'
                             'Possible values 0 or 1.')
    parser.add_argument('--training-epochs', type=int, default=1000,
                        help='Number of epochs to train the mlp classifier')


    return parser.parse_args()


def save_evaluation_results(classifier, dirname, classifier_factory):
    accuracy, precision, recall, fscore, y_true, y_pred = classifier.evaluate(
        classifier_factory=classifier_factory,
    )
    evaluation_results = pandas.DataFrame(
        columns=['accuracy', 'class', 'precision', 'recall', 'fscore'])
    evaluation_results = evaluation_results.append({'accuracy': accuracy},
                                                   ignore_index=True)
    for cls_idx, cls in enumerate(classifier.classes[1]):
        evaluation_results = evaluation_results.append({
            'class': cls,
            'precision': precision[cls_idx],
            'recall': recall[cls_idx],
            'fscore': fscore[cls_idx],
            'support': (y_true == cls_idx).sum()
        }, ignore_index=True)
    evaluation_results.to_csv(os.path.join(dirname, 'evaluation_results.csv'),
                              index=False)

    predictions = pandas.DataFrame(numpy.vstack([y_true, y_pred]).T,
                                   columns=['true', 'prediction'])
    predictions.to_csv(os.path.join(dirname, 'evaluation_predictions.csv'),
                       index=False)



def main():
    """Main function of script."""
    args = read_arguments()
    classifier = double_step_classifier.DoubleStepClassifier(
        dataset_class=HandcraftedFeaturesDataset, use_trained=args.use_trained)

    classifier.load_from_files(
        dataset_filepath=args.input_matrix_file,
        labels_filepath=args.labels_filepath,
        labels=((3, 'wordnet'), (4, 'uri')),
        indices_filepath=args.indices)
    factory = None

    if args.classifier == 'mlp':
        factory = double_step_classifier.MLPFactory(
            results_save_path=args.results_dirname,
            training_epochs=args.training_epochs,
            dropout_ratio=args.dropout_ratio, num_layers=args.num_layers)

    elif args.classifier == 'heuristic':
        factory = heuristic_classifier.HeuristicClassifierFactory(
            args.features_filename)
    elif args.classifier == 'lr':
        factory = logistic_regression.LRClassifierFactory(
            save_models=True, results_save_path=args.results_dirname)
    elif args.classifier == 'random':
        factory = random_classifier.RandomClassifierFactory()
    elif args.classifier == 'knn':
        factory = nn_classifier.NNeighborsClassifierFactory(
            args.features_filename, args.results_dirname)

    logging.info('Starting evaluation')
    save_evaluation_results(classifier, args.results_dirname, factory)

    if args.classifier == 'mlp':
        classifier.close_open_sessions()
    logging.info('All operations completed')


if __name__ == '__main__':
    main()

