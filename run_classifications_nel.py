# -*- coding: utf-8 -*-
"""Script to train a classifier for NE linking using a double step classifier.

python run_classifications_nel.py ../data/legal_sampled/filtered_handcreafter_matrix.npz  --labels_filepath ../data/legal_sampled/labels.pickle --results_dirname ../results/double_step_classifier/sample/ --indices ../data/legal_sampled/indices.npz --classifier 'heuristic' --features_filename ../data/legal_sampled/filtered_features_file_path --entities_filename ../data/mappings/all_uris_labels.pickle
"""

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import jsonlines
import logging
import pickle

from collections import defaultdict

logging.basicConfig(level=logging.INFO)

from wikipedianer.classification import double_step_classifier
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
                             'and heuristic.')
    parser.add_argument('--features_filename', type=str, default=None,
                        help='Path of file with the filtered features. Only '
                             'for heuristic classifier.')
    parser.add_argument('--entities_filename', type=str, default=None,
                        help='Path of jsonlines file with the entity labels. '
                             'Only for heuristic classifier.')


    return parser.parse_args()


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
    if args.classifier == 'mlp':
        factory = double_step_classifier.MLPFactory(
            results_save_path=args.results_dirname, training_epochs=100,
            layers=[1000])
        classifier.train(classifier_factory=factory)
        classifier.save_to_file(results_dirname=args.results_dirname)
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
        classifier.train(classifier_factory=factory)
        classifier.save_to_file(results_dirname=args.results_dirname)
        metrics = classifier.evaluate()

if __name__ == '__main__':
    main()

