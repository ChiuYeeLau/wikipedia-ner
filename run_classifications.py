# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle as pickle
import numpy as np
import os
import sys
import tensorflow as tf

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from utils import LABELS_REPLACEMENT
from wikipedianer.classification.mlp import MultilayerPerceptron


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=unicode)
    parser.add_argument("labels", type=unicode)
    parser.add_argument("mappings", type=unicode)
    parser.add_argument("indices_dir", type=unicode)
    parser.add_argument("results_dir", type=unicode)
    parser.add_argument("saves_dir", type=unicode)
    parser.add_argument("--mappings_kind", type=unicode, default='NEU', nargs='+')
    parser.add_argument("--experiment_kind", type=unicode, default='legal')
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--loss_report", type=int, default=50)
    parser.add_argument("--layers", type=lambda x: map(int, x.split(',')), nargs='+', default=[[12000, 9000]])
    parser.add_argument("--dynamic_layers", type=int, nargs='+', default=None)
    parser.add_argument("--dropout_ratios", type=float, default=None, nargs='+')

    args = parser.parse_args()

    args.mappings_kind = [args.mappings_kind] if isinstance(args.mappings_kind, unicode) else args.mappings_kind

    for mapping_kind in args.mappings_kind:
        if mapping_kind not in LABELS_REPLACEMENT[args.experiment_kind]:
            print('Not a valid replacement {}'.format(mapping_kind), file=sys.stderr)
            sys.exit(1)

    if len(args.mappings_kind) != len(args.layers):
        print('Layers and mappings don\'t have the same amount of items', file=sys.stderr)
        sys.exit(1)

    args.dynamic_layers = [args.dynamic_layers] if isinstance(args.dynamic_layers, int) else args.dynamic_layers

    if args.dynamic_layers is not None and len(args.dynamic_layers) != len(args.layers) - 1:
        print('The number of dynamic layers must be one less than the number of layers', file=sys.stderr)
        sys.exit(1)

    args.dropout_ratios = [args.dropout_ratios] if isinstance(args.dropout_ratios, int) else args.dropout_ratios

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
            "_".join(args.mappings_kind[:idx+1])), "_".join([unicode(l) for l in args.layers[idx]])
        )
        experiments_name.append(experiment_name)

        print('Running experiment: {}'.format(experiment_name), file=sys.stderr)

        print('Replacing the labels', file=sys.stderr)
        replacement_function = LABELS_REPLACEMENT[args.experiment_kind][mapping_kind]
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

        dynamic_layer = args.dynamic_layers.pop(0) if len(args.dynamic_layers) > 0 else None

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
                                       save_model=save_model, dropout_ratios=args.dropout_ratios,
                                       dynamic_layer=dynamic_layer)

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
