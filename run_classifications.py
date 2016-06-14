# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle as pickle
import numpy as np
import re
import sys

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from wikipedianer.classification.mlp import MultilayerPerceptron


def ner_label_replace(labels, mappings):
    for label in labels:
        if label.startswith('O'):
            yield 'O'
        else:
            yield 'I'


def ne_person_label_replace(labels, mappings):
    for label in labels:
        label = re.sub(r'^[BI]-', '', label)
        if 'no_person' in mappings.get(label, set()):
            yield 'no_person'
        elif 'wordnet_person_100007846' in mappings.get(label, set()):
            yield 'wordnet_person_100007846'
        else:
            yield 'O'


def ne_category_label_replace(labels, mappings):
    for label in labels:
        label = re.sub(r'^[BI]-', '', label)
        if 'wordnet_movie_106613686' in mappings.get(label, set()):
            yield 'wordnet_movie_106613686'
        elif 'wordnet_soundtrack_104262969' in mappings.get(label, set()):
            yield 'wordnet_soundtrack_104262969'
        elif 'wordnet_actor_109765278' in mappings.get(label, set()):
            yield 'wordnet_actor_109765278'
        elif 'wordnet_film_director_110088200' in mappings.get(label, set()):
            yield 'wordnet_film_director_110088200'
        elif 'wordnet_film_maker_110088390' in mappings.get(label, set()):
            yield 'wordnet_film_maker_110088390'
        else:
            yield 'O'


def ne_uri_label_replace(labels, mappings):
    for label in labels:
        yield re.sub(r"^B-", "I-", label)


labels_replacements = dict(
    NER=ner_label_replace,
    NEP=ne_person_label_replace,
    NEC=ne_category_label_replace,
    NEU=ne_uri_label_replace
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=unicode)
    parser.add_argument("labels", type=unicode)
    parser.add_argument("mappings", type=unicode)
    parser.add_argument("mapping_kind", type=unicode)
    parser.add_argument("indices", type=unicode)
    parser.add_argument("logs_dir", type=unicode)
    parser.add_argument("results_dir", type=unicode)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--loss_report", type=int, default=50)
    parser.add_argument("--layers", type=int, nargs='+')

    args = parser.parse_args()

    if args.mapping_kind not in labels_replacements:
        print('Not a valid replacement', file=sys.stderr)
        sys.exit(1)

    print('Loading dataset from file {}'.format(args.dataset), file=sys.stderr)

    dataset = np.load(args.dataset)
    dataset = csr_matrix((dataset['data'], dataset['indices'], dataset['indptr']), shape=dataset['shape'])

    print('Loading labels from file {}'.format(args.labels), file=sys.stderr)
    with open(args.labels, 'rb') as f:
        labels = pickle.load(f)

    print('Loading class mappings from file {}'.format(args.mappings), file=sys.stderr)
    with open(args.mappings, 'rb') as f:
        class_mappings = pickle.load(f)

    print('Replacing the labels', file=sys.stderr)
    replacement_function = labels_replacements[args.mapping_kind]
    labels = list(replacement_function(labels, class_mappings))

    print('Loading indices for train, test and validation from file {}'.format(args.indices), file=sys.stderr)
    indices = np.load(args.indices)

    print('Filtering dataset and labels according to indices', file=sys.stderr)
    dataset = dataset[indices['filtered_indices']]
    labels = np.array(labels)[indices['filtered_indices']]

    print('Normalizing dataset', file=sys.stderr)
    dataset = normalize(dataset.astype(np.float32), norm='max', axis=0)

    print('Creating multilayer perceptron', file=sys.stderr)
    mlp = MultilayerPerceptron(dataset, labels, indices['train_indices'], indices['test_indices'],
                               indices['validation_indices'], args.logs_dir, args.results_dir, args.layers,
                               args.learning_rate, args.epochs, args.batch_size, args.loss_report)

    print('Training the classifier', file=sys.stderr)
    mlp.train()

    print('Finished the experiment', file=sys.stderr)
