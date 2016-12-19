# -*- coding: utf-8 -*-
"""Script to train a classifier for NE linking using a double step classifier.
"""

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import logging

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
    factory = double_step_classifier.MLPFactory(
        results_save_path=args.results_dirname, training_epochs=100,
        layers=[1000])
    classifier.train(classifier_factory=factory)
    classifier.save_to_file(results_dirname=args.results_dirname)



if __name__ == '__main__':
    main()

