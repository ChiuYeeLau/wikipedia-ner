# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import numpy as np
import os
import sys
import tensorflow as tf

from wikipedianer.classification.mlp import MultilayerPerceptron
from wikipedianer.dataset import HandcraftedFeaturesDataset, WordVectorsDataset
from wikipedianer.pipeline.util import CL_ITERATIONS


def run_classifier(dataset_path, labels_path, indices_path, results_save_path, pre_trained_weights_save_path,
                   cl_iterations, word_vectors_path=None, layers=list(), dropout_ratios=list(), save_models=list(),
                   completed_iterations=list(), learning_rate=0.01, epochs=10000, batch_size=2100, loss_report=250,
                   batch_normalization=False, debug_word_vectors=False):
    if word_vectors_path or debug_word_vectors:
        dataset = WordVectorsDataset(dataset_path, labels_path, indices_path, word_vectors_path, dtype=np.float32,
                                     debug=debug_word_vectors)
    else:
        dataset = HandcraftedFeaturesDataset()
        dataset.load_from_files(dataset_path, labels_path, indices_path, dtype=np.int32)

    experiments_names = []

    for iteration in completed_iterations:
        experiment_name = '%s_%s' % ('_'.join(CL_ITERATIONS[:iteration+1]), '_'.join([str(l) for l in layers]))
        experiments_names.append(experiment_name)

    for iteration in cl_iterations:
        if iteration in completed_iterations:
            print('Skipping completed iterations %s' % CL_ITERATIONS[iteration], file=sys.stderr, flush=True)
            continue

        if len(cl_iterations) > 1:
            experiment_name = '%s_%s' % ('_'.join(CL_ITERATIONS[:iteration+1]), '_'.join([str(l) for l in layers]))
        else:
            experiment_name = '%s_%s' % (CL_ITERATIONS[iteration], '_'.join([str(l) for l in layers]))

        experiments_names.append(experiment_name)

        print('Running experiment: %s' % experiment_name, file=sys.stderr, flush=True)

        if len(experiments_names) > 1:
            print('Loading previous weights and biases', file=sys.stderr, flush=True)
            pre_weights = np.load(os.path.join(pre_trained_weights_save_path, '%s_weights.npz' % experiments_names[-2]))
            pre_biases = np.load(os.path.join(pre_trained_weights_save_path, '%s_biases.npz' % experiments_names[-2]))
        else:
            pre_weights = None
            pre_biases = None

        save_model = save_models[iteration]

        with tf.Graph().as_default() as g:
            tf.set_random_seed(1234)

            do_ratios = dropout_ratios[:] if dropout_ratios is not None else None

            print('Creating multilayer perceptron', file=sys.stderr, flush=True)
            mlp = MultilayerPerceptron(dataset=dataset, pre_trained_weights_save_path=pre_trained_weights_save_path,
                                       results_save_path=results_save_path, experiment_name=experiment_name,
                                       cl_iteration=iteration, layers=layers, learning_rate=learning_rate,
                                       training_epochs=epochs, batch_size=batch_size, loss_report=loss_report,
                                       pre_weights=pre_weights, pre_biases=pre_biases, save_model=save_model,
                                       dropout_ratios=do_ratios, batch_normalization=batch_normalization)

            print('Training the classifier', file=sys.stderr, flush=True)
            mlp.train()

        # Releasing some memory
        del pre_weights
        del pre_biases
        del mlp
        del g

        print('Finished experiment %s' % experiment_name, file=sys.stderr, flush=True)

    print('Finished all the experiments', file=sys.stderr, flush=True)
