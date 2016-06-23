# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle as pickle
import numpy as np
import os
import re
import sys
import tensorflow as tf

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
        if 'wordnet_law_108441203' in mappings.get(label, set()):
            yield 'wordnet_law_108441203'
        elif 'wordnet_law_100611143' in mappings.get(label, set()):
            yield 'wordnet_law_100611143'
        elif 'wordnet_law_106532330' in mappings.get(label, set()):
            yield 'wordnet_law_106532330'
        elif 'wordnet_legal_document_106479665' in mappings.get(label, set()):
            yield 'wordnet_legal_document_106479665'
        elif 'wordnet_due_process_101181475' in mappings.get(label, set()):
            yield 'wordnet_due_process_101181475'
        elif 'wordnet_legal_code_106667792' in mappings.get(label, set()):
            yield 'wordnet_legal_code_106667792'
        elif 'wordnet_criminal_record_106490173' in mappings.get(label, set()):
            yield 'wordnet_criminal_record_106490173'
        elif 'wordnet_legal_power_105198427' in mappings.get(label, set()):
            yield 'wordnet_legal_power_105198427'
        elif 'wordnet_jurisdiction_108590369' in mappings.get(label, set()):
            yield 'wordnet_jurisdiction_108590369'
        elif 'wordnet_judiciary_108166318' in mappings.get(label, set()):
            yield 'wordnet_judiciary_108166318'
        elif 'wordnet_pleading_106559365' in mappings.get(label, set()):
            yield 'wordnet_pleading_106559365'
        elif 'wordnet_court_108329453' in mappings.get(label, set()):
            yield 'wordnet_court_108329453'
        elif 'wordnet_judge_110225219' in mappings.get(label, set()):
            yield 'wordnet_judge_110225219'
        elif 'wordnet_adjudicator_109769636' in mappings.get(label, set()):
            yield 'wordnet_adjudicator_109769636'
        elif 'wordnet_lawyer_110249950' in mappings.get(label, set()):
            yield 'wordnet_lawyer_110249950'
        elif 'wordnet_party_110402824' in mappings.get(label, set()):
            yield 'wordnet_party_110402824'
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
    parser.add_argument("indices_dir", type=unicode)
    parser.add_argument("results_dir", type=unicode)
    parser.add_argument("saves_dir", type=unicode)
    parser.add_argument("--mappings_kind", type=unicode, default='NEU', nargs='+')
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--loss_report", type=int, default=50)
    parser.add_argument("--layers", type=lambda x: map(int, x.split(',')), nargs='+', default=[[12000, 9000, 7000]])

    args = parser.parse_args()

    args.mappings_kind = [args.mappings_kind] if isinstance(args.mappings_kind, unicode) else args.mappings_kind

    for mapping_kind in args.mappings_kind:
        if mapping_kind not in labels_replacements:
            print('Not a valid replacement {}'.format(mapping_kind), file=sys.stderr)
            sys.exit(1)

    if len(args.mappings_kind) != len(args.layers):
        print('Layers and mappings don\'t have the same amount of items')
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

    experiments_name = []

    for idx, mapping_kind in enumerate(args.mappings_kind):
        experiment_name = "{}_{}".format("{}".format(
            "_".join(args.mappings_kind[:idx] + [mapping_kind])), "_".join([unicode(l) for l in args.layers[idx]])
        )
        experiments_name.append(experiment_name)

        print('Running experiment: {}'.format(experiment_name), file=sys.stderr)

        print('Replacing the labels', file=sys.stderr)
        replacement_function = labels_replacements[mapping_kind]
        experiment_labels = list(replacement_function(labels, class_mappings))

        print('Loading indices for train, test and validation', file=sys.stderr)
        indices = np.load(os.path.join(args.indices_dir, "{}_indices.npz".format(mapping_kind)))

        print('Filtering dataset and labels according to indices', file=sys.stderr)
        experiment_dataset = dataset[indices['filtered_indices']]
        experiment_labels = np.array(experiment_labels)[indices['filtered_indices']]

        print('Normalizing dataset', file=sys.stderr)
        experiment_dataset = normalize(experiment_dataset.astype(np.float32), norm='max', axis=0)

        if len(experiments_name) > 1:
            print('Loading previous weights and biases', file=sys.stderr)
            pre_weights = np.load(os.path.join(args.saves_dir, '{}_weights.npz'.format(experiments_name[-2])))
            pre_biases = np.load(os.path.join(args.saves_dir, '{}_biases.npz'.format(experiments_name[-2])))
        else:
            pre_weights = None
            pre_biases = None

        save_model = True if idx >= len(args.mappings_kind) - 2 else False

        with tf.Graph().as_default() as g:
            tf.set_random_seed(1234)

            print('Creating multilayer perceptron', file=sys.stderr)
            mlp = MultilayerPerceptron(dataset=experiment_dataset, labels=experiment_labels,
                                       train_indices=indices['train_indices'], test_indices=indices['test_indices'],
                                       validation_indices=indices['validation_indices'], saves_dir=args.saves_dir,
                                       results_dir=args.results_dir, experiment_name=experiment_name,
                                       layers=args.layers[idx], learning_rate=args.learning_rate,
                                       training_epochs=args.epochs, batch_size=args.batch_size,
                                       loss_report=args.loss_report, pre_weights=pre_weights, pre_biases=pre_biases,
                                       save_model=save_model)

            print('Training the classifier', file=sys.stderr)
            mlp.train()

        # Releasing some memory
        del pre_weights
        del pre_biases
        del experiment_dataset
        del experiment_labels
        del indices
        del mlp
        del g

        print('Finished experiment {}'.format(experiment_name), file=sys.stderr)

    print('Finished all the experiments', file=sys.stderr)
