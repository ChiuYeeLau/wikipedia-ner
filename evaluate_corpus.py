# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle
import numpy as np
import os
import sys
import tensorflow as tf
from scipy.sparse import csr_matrix
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=unicode)
    parser.add_argument('classes', type=unicode)
    parser.add_argument('model', type=unicode)
    parser.add_argument('words', type=unicode)
    parser.add_argument('results', type=unicode)
    parser.add_argument('--layers', type=int, nargs='+', default=[12000, 10000])
    parser.add_argument('--batch_size', type=int, default=2000)

    args = parser.parse_args()

    print('Loading dataset from file {}'.format(args.dataset), file=sys.stderr)
    dataset = np.load(args.dataset)
    dataset = csr_matrix((dataset['data'], dataset['indices'], dataset['indptr']), shape=dataset['shape'])

    print('Loading classes from file {}'.format(args.classes), file=sys.stderr)
    with open(args.classes, 'rb') as f:
        classes = np.array(cPickle.load(f))

    layers_size = [args.layers] if isinstance(args.layers, int) else args.layers
    biases_names = "biases" if os.path.basename(args.model).startswith("NEU") else None

    input_size = dataset.shape[1]
    output_size = classes.shape[0]
    y_pred = np.zeros(dataset.shape[0], dtype=np.int32)

    with tf.Graph().as_default() as g:
        X = tf.placeholder(tf.float32, shape=(None, input_size), name='X')
        layers = [X]
        weights = []
        biases = []

        print('Building neural network with architecture: {}'
              .format(' -> '.join(map(str, [input_size] + layers_size + [output_size]))),
              file=sys.stderr)

        # Create the layers
        for layer_idx, (size_prev, size_current) in enumerate(zip([input_size] + layers_size, layers_size)):
            print('Creating hidden layer {:02d}: {} -> {}'.format(layer_idx, size_prev, size_current), file=sys.stderr)

            layer_name = 'hidden_layer_{:02d}'.format(layer_idx)

            with tf.name_scope(layer_name):
                weights_variable = tf.Variable(tf.zeros((size_prev, size_current), dtype=tf.float32), name='weights')
                biases_variable = tf.Variable(tf.zeros([size_current], dtype=tf.float32), name=biases_names)

                layer = tf.nn.relu(tf.matmul(layers[-1], weights_variable) + biases_variable)

                weights.append(weights_variable)
                biases.append(biases_variable)
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

    print('Loading words of corpus from file {}'.format(args.words), file=sys.stderr)
    with open(args.words, 'rb') as f:
        words = cPickle.load(f)

    print('Saving resulting corpus to dir {}'.format(args.results), file=sys.stderr)
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

    print('All finished', file=sys.stderr)

