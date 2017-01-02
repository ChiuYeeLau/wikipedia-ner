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
from wikipedianer.classification import logistic_regression
from wikipedianer.dataset import HandcraftedFeaturesDataset



def read_arguments():
    """Parses the arguments from the stdin and returns an object."""
    parser = argparse.ArgumentParser()
    parser.add_argument('input_matrix_file', type=str,
                        help='Path of file with the numpy matrix used for'
                             'training')
    parser.add_argument('--labels_filepath', '-hl,', type=str,
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
                             'lr and heuristic.')
    parser.add_argument('--features_filename', type=str, default=None,
                        help='Path of file with the filtered features. Only '
                             'for heuristic classifier.')
    parser.add_argument('--entities_filename', type=str, default=None,
                        help='Path of jsonlines file with the entity labels. '
                             'Only for heuristic classifier.')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Do not train the classifier, only perform'
                             'evaluation')
    parser.add_argument('--dropout-ratio', type=float, default=0.0,
                        help='The dropout ratio for the classifier.')

    return parser.parse_args()


def save_evaluation_results(classifier, dirname, classifier_factory):
    accuracy, precision, recall, fscore, y_true, y_pred = classifier.evaluate(
        classifier_factory=classifier_factory
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
            'fscore': fscore[cls_idx]
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
        dataset_class=HandcraftedFeaturesDataset)

    classifier.load_from_files(
        dataset_filepath=args.input_matrix_file,
        labels_filepath=args.labels_filepath,
        labels=((3, 'wordnet'), (4, 'uri')),
        indices_filepath=args.indices)
    factory = None

    if args.classifier == 'mlp':
        factory = double_step_classifier.MLPFactory(
            results_save_path=args.results_dirname, training_epochs=100,
            dropout_ratio=args.dropout_ratio)

    elif args.classifier == 'heuristic':
        # Read the entities
        entities_database = defaultdict(set)
        with jsonlines.open(args.entities_filename) as reader:
            for entity in reader:
                uri = entity['uri']
                labels = entity['labels']
                for label in labels:
                    for word in label.split():
                        entities_database[word].add(uri)

        # Read the features
        with open(args.features_filename, 'rb') as feature_file:
            features = pickle.load(feature_file)
            # Get indices for each current token
        token_features = {
            index: x.split('token:current=')[1]
            for index, x in enumerate(features)
            if x.startswith('token:current')}
        factory = double_step_classifier.HeuristicClassifierFactory(
            entities_database, token_features)
    elif args.classifier == 'lr':
        factory = logistic_regression.LRClassifierFactory(
            save_models=True, results_save_path=args.results_dirname)

    if not args.evaluate_only:
        classifier.train(classifier_factory=factory)
        classifier.save_to_file(results_dirname=args.results_dirname)

    logging.info('Starting evaluation')
    save_evaluation_results(classifier, args.results_dirname, factory)

    if args.classifier == 'mlp':
        classifier.close_open_sessions()
    logging.info('All operations completed')


if __name__ == '__main__':
    main()

