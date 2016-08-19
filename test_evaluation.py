# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle
import numpy as np
import sys
import tensorflow as tf
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=unicode)
    parser.add_argument('labels', type=unicode)
    parser.add_argument('indices', type=unicode)
    parser.add_argument('model', type=unicode)
    parser.add_argument('results', type=unicode)
    parser.add_argument('--layers', type=int, nargs='+')
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--word_vectors', action='store_true')
    parser.add_argument('--batch_normalization', action='store_true')

    args = parser.parse_args()

    print('Loading dataset from file {}'.format(args.dataset), file=sys.stderr)
    if args.word_vectors:
        dataset = np.load(args.dataset)['dataset']
    else:
        dataset = np.load(args.dataset)
        dataset = csr_matrix((dataset['data'], dataset['indices'], dataset['indptr']), shape=dataset['shape'])

        print('Normalizing dataset', file=sys.stderr)
        dataset = normalize(dataset.astype(np.float32), norm='max', axis=0)

    print('Loading labels from file {}'.format(args.labels), file=sys.stderr)
    with open(args.labels, 'rb') as f:
        labels = np.array(cPickle.load(f))

    print('Loading indices from file {}'.format(args.indices), file=sys.stderr)
    indices = np.load(args.indices)

    print('Getting test dataset', file=sys.stderr)
    dataset = dataset[indices['filtered_indices']]['test_indices']
    labels = labels[indices['filtered_indices']]['test_indices']

    layers_size = [args.layers] if isinstance(args.layers, int) else args.layers

    input_size = dataset.shape[1]
    output_size = np.unique(labels).shape[0]
    y_pred = np.zeros(dataset.shape[0], dtype=np.int32)

    with tf.Graph().as_default() as g:
        X = tf.placeholder(tf.float32, shape=(None, input_size), name='X')
        layers = [X]

        print('Building neural network with architecture: {}'
              .format(' -> '.join(map(str, [input_size] + layers_size + [output_size]))),
              file=sys.stderr)

        # Create the layers
        for layer_idx, (size_prev, size_current) in enumerate(zip([input_size] + layers_size, layers_size)):
            print('Creating hidden layer {:02d}: {} -> {}'.format(layer_idx, size_prev, size_current), file=sys.stderr)

            layer_name = 'hidden_layer_{:02d}'.format(layer_idx)

            with tf.name_scope(layer_name):
                weights_variable = tf.Variable(tf.zeros((size_prev, size_current), dtype=tf.float32), name='weights')
                biases_variable = tf.Variable(tf.zeros([size_current], dtype=tf.float32), name="biases")

                layer = tf.nn.relu(tf.matmul(layers[-1], weights_variable) + biases_variable)

                if args.batch_normalization:
                    mean, var = tf.nn.moments(layer, axes=[0])
                    layer = tf.nn.batch_normalization(layer, mean, var, None, None, 1e-10)

                layers.append(layer)

        # The last layer is softmax
        with tf.name_scope('softmax_layer'):
            print('Creating softmax layer: {} -> {}'.format(layers_size[-1], output_size), file=sys.stderr)
            weights_variable = tf.Variable(tf.zeros([layers_size[-1], output_size], dtype=tf.float32), name='weights')
            biases_variable = tf.Variable(tf.zeros([output_size]), name='biases')
            y_logits = tf.matmul(layers[-1], weights_variable) + biases_variable

        classification = tf.argmax(tf.nn.softmax(y_logits), 1, name='y_predictions')

        saver = tf.train.Saver()

        print('Starting session for classification', file=sys.stderr)
        with tf.Session() as sess:
            print('Loading model from file {}'.format(args.model), file=sys.stderr)
            saver.restore(sess, args.model)

            print('Running classification for dataset {}'.format(args.dataset), file=sys.stderr)
            for step in tqdm(np.arange(dataset.shape[0], step=args.batch_size)):
                dataset_chunk = dataset[step:min(step+args.batch_size, dataset.shape[0])]
                feed_dict = {
                    X: dataset_chunk.toarray() if hasattr(dataset_chunk, 'toarray') else dataset_chunk
                }

                y_pred[step:min(step+args.batch_size, dataset.shape[0])] = sess.run(classification, feed_dict=feed_dict)

    print('Saving results to file {}'.format(args.results), file=sys.stderr)
    np.savetxt(args.results, y_pred, fmt='%d'.encode('utf-8'))

    print('All finished', file=sys.stderr)
