# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
try:
    import cPickle as pickle
except ImportError:
    import pickle
import logging
import numpy as np
import pandas
import sys
import os
from wikipedianer.pipeline import util

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm
from wikipedianer.dataset import HandcraftedFeaturesDataset, WordVectorsDataset
from wikipedianer.classification.mlp import MultilayerPerceptron
from wikipedianer.classification.double_step_classifier import (
    DoubleStepClassifier, MLPFactory)

logging.basicConfig(level=logging.INFO)

def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                        help='Path to the file with the dataset. For the'
                             'HandcraftedFeatureDataset, a sparse matrix is '
                             'expected. For the WordVectorDataset, a pickled'
                             'file is expected.')
    parser.add_argument('classes', type=str)
    parser.add_argument('model', type=str, help='Path to the model(s) to load.')
    parser.add_argument('words', type=str)
    parser.add_argument('--task', type=str, default='URI',
                        help='Task to use as label. Possible values are:'
                             'YAGO, NER, ENTITY, LKIF, URI')
    parser.add_argument('--results_save_path', type=str,
                        help='Path to directory where to save the results.')
    parser.add_argument('--layers', type=int, nargs='+')
    parser.add_argument('--classifier', type=str, default='mlp',
                        help='Name of the classifier to evaluate. Default is'
                             'mlp, possible values are mlp or double-step.')
    parser.add_argument('--hl_predictions', type=str,
                        help='Path to file with the high level predictions to'
                             'use with the DoubleStepClassifier.')
    parser.add_argument('--word_vectors', type=str, default=None,
                        help='Path to file with the word_vector model. If none'
                             'is provided, a HandcraftedFeatureDataset will be'
                             'used.')
    parser.add_argument("--batch_normalization", action='store_true')

    return parser.parse_args()


def create_dataset(args):
    """Returns an instance of Dataset according to the parameters in args."""
    logging.info('Loading classes from file {}'.format(args.classes))
    with open(args.classes, 'rb') as f:
        raw_classes = pickle.load(f)
        classes = np.array([raw_classes[task][0]
                            for task in util.CL_ITERATIONS])

    logging.info('Loading dataset from file {}'.format(args.dataset))
    if args.word_vectors is not None:
        dataset = WordVectorsDataset(word_vectors_path=args.word_vectors,
                                     dtype=np.float32)
        matrix = pickle.load(args.dataset)
    else:
        matrix = np.load(args.dataset)
        matrix = csr_matrix((matrix['data'], matrix['indices'],
                             matrix['indptr']), shape=matrix['shape'])

        logging.info('Normalizing dataset')
        matrix = normalize(matrix.astype(np.float32), norm='max', axis=0)
        dataset = HandcraftedFeaturesDataset(dtype=np.int32)

    dataset.load_for_evaluation(matrix, classes)

    return dataset


def main():
    args = read_arguments()
    logging.info('Creating dataset.')
    dataset = create_dataset(args)

    layers_size = [args.layers] if isinstance(args.layers, int) else args.layers
    iteration = util.CL_ITERATIONS.index(args.task)
    if iteration == -1:
        logging.info('Task is not valid')
        return

    classifier = None
    if args.classifier == 'mlp':
        assert args.task != 'URI'
        batch_size = min(2000, dataset.num_examples('test'))
        classifier = MultilayerPerceptron(
            dataset, results_save_path=args.results_save_path,
            experiment_name='evaluation', layers=layers_size,
            save_model=True, cl_iteration=iteration,
            batch_size=batch_size, ignore_batch_size=True)
        y_pred = classifier.predict(dataset_name='test', restore=True,
                                    model_name=args.model)
    elif args.classifier == 'double-step':
        assert args.task == 'URI'
        classifier = DoubleStepClassifier(use_trained=True, dtype=np.int32,
                                          dataset_class=type(dataset))
        dataset.classes = dataset.classes[-2:]  # Keep only the last 2 classes
        iteration = 1
        classifier.read_from_file(args.model)
        classifier.load_dataset(dataset)
        factory = MLPFactory(results_save_path=args.model, num_layers=1)
        hl_predictions = np.array(pandas.read_csv(args.hl_predictions)['0'])
        y_pred = classifier.predict(
            dataset_name='test', classifier_factory=factory,
            predicted_high_level_labels=hl_predictions)
    if not classifier:
        logging.info('Classifier not created')
        return

    logging.info('Loading words of corpus from file {}'.format(args.words))
    with open(args.words, 'rb') as f:
        words = pickle.load(f)

    logging.info('Saving resulting corpus to dir {}'.format(
        args.results_save_path))
    # Save numeric predictions
    output_filename = os.path.join(
        args.results_save_path, 'predictions_{}.csv'.format(args.task))
    pandas.DataFrame(y_pred).to_csv(output_filename, index=None)

    # Save predictions for evaluation
    output_filename = os.path.join(
        args.results_save_path, 'readable_predictions_{}.csv'.format(args.task))
    with open(output_filename, 'wb') as outfile:
        for idx, (word_idx, token, tag, is_doc_start) in tqdm(enumerate(words)):
            word_label = dataset.classes[iteration][int(y_pred[idx])]

            if idx > 0 and word_idx == 0:
                outfile.write('\n'.encode('utf-8'))

            row = '{}\t{}\t{}\t{}\n'.format(word_idx, token, tag, word_label)
            outfile.write(row.encode('utf-8'))

    print('All finished', file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()