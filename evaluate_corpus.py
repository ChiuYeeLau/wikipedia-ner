# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import sys
import utils

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm
from wikipedianer.dataset import HandcraftedFeaturesDataset, WordVectorsDataset
from wikipedianer.classification.mlp import MultilayerPerceptron
from wikipedianer.classification.double_step_classifier import DoubleStepClassifier


def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                        help='Path to the file with the dataset. For the'
                             'HandcraftedFeatureDataset, a sparse matrix is '
                             'expected. For the WordVectorDataset, a pickled'
                             'file is expected.')
    parser.add_argument('classes', type=str)
    parser.add_argument('model', type=str,
                        help='Path to the model to load.')
    parser.add_argument('words', type=str)
    parser.add_argument('--task', type=str, default='URI',
                        help='Task to use as label. Possible values are:'
                             'YAGO, NER, ENTITY, LKIF, URI')
    parser.add_argument('--results_save_path', type=str,
                        help='Path to directory where to save the results.')
    parser.add_argument('--layers', type=int, nargs='+')
    parser.add_argument('--classifier', type=str, default='mlp',
                        help='Name of the classifier to evaluate. Default is'
                             'mlp, possible values are mlp or nel.')
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--word_vectors', type=str, default=None,
                        help='Path to file with the word_vector model. If none'
                             'is provided, a HandcraftedFeatureDataset will be'
                             'used.')
    parser.add_argument("--batch_normalization", action='store_true')

    return parser.parse_args()


def create_dataset(args):
    """Returns an instance of Dataset according to the parameters in args."""
    print('Loading classes from file {}'.format(args.classes), file=sys.stderr,
          flush=True)
    with open(args.classes, 'rb') as f:
        classes = np.array(pickle.load(f)[args.task][0])

    print('Loading dataset from file {}'.format(args.dataset), file=sys.stderr,
          flush=True)
    if args.word_vectors is not None:
        dataset = WordVectorsDataset(word_vectors_path=args.word_vectors,
                                     dtype=np.float32)
        matrix = pickle.load(args.dataset)
    else:
        matrix = np.load(args.dataset)
        matrix = csr_matrix((matrix['data'], matrix['indices'],
                             matrix['indptr']), shape=matrix['shape'])

        print('Normalizing dataset', file=sys.stderr, flush=True)
        matrix = normalize(matrix.astype(np.float32), norm='max', axis=0)
        dataset = HandcraftedFeaturesDataset(dtype=np.int32)

    dataset.load_for_evaluation(matrix, classes)

    return dataset


if __name__ == "__main__":
    args = read_arguments()
    dataset = create_dataset()

    layers_size = [args.layers] if isinstance(args.layers, int) else args.layers
    iteration = utils.CL_ITERATIONS[args.task]

    input_size = dataset.input_size
    output_size = dataset.output_size(iteration)
    y_pred = np.zeros(dataset.num_examples('test'), dtype=np.int32)

    if args.classifier == 'mlp':
        assert args.task != 'URI'
        batch_size = min(2000, dataset.num_examples('test'))
        classifier = MultilayerPerceptron(
            dataset, results_save_path=args.results_save_path,
            experiment_name='evaluation', layers=layers_size,
            save_model=False, cl_iteration=iteration,
            batch_size=batch_size)
    elif args.classifier == 'nel':
        assert args.task == 'URI'
        classifier = DoubleStepClassifier(use_trained=False, dtype=np.int32)
        classifier


    accuracy, precision, recall, fscore, y_true, y_pred = classifier.evaluate()

    print('Loading words of corpus from file {}'.format(args.words),
          file=sys.stderr, flush=True)
    with open(args.words, 'rb') as f:
        words = pickle.load(f)

    print('Saving resulting corpus to dir {}'.format(args.results), file=sys.stderr, flush=True)
    with open(args.results, 'w') as f:
        for idx, (word_idx, token, tag, is_doc_start) in tqdm(enumerate(words)):
            word_label = classes[int(y_pred[idx])]

            if idx > 0 and word_idx == 0:
                print(''.encode('utf-8'), file=f)
                if is_doc_start:
                    print('-'.encode('utf-8') * 100, file=f)
                    print('-'.encode('utf-8') * 100, file=f)

            doc_title = 'DOCUMENT START' if is_doc_start else ''

            print('{}\t{}\t{}\t{}\t{}'.format(word_idx, token, tag, word_label, doc_title).encode('utf-8'), file=f)

    print('All finished', file=sys.stderr, flush=True)

