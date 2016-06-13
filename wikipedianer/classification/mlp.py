# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import sys
from sklearn.metrics import precision_score, recall_score
from .base import BaseClassifier


class MultilayerPerceptron(BaseClassifier):
    def __init__(self, dataset, labels, train_indices, test_indices, validation_indices, logs_dir, results_dir,
                 layers, learning_rate=0.01, training_epochs=1000, batch_size=100):
        super(MultilayerPerceptron, self).__init__(dataset, labels, train_indices, test_indices, validation_indices)

        assert batch_size <= self.train_dataset.shape[0]

        self.X = tf.placeholder(tf.float32, shape=(batch_size, self.input_size), name='X')
        self.y = tf.placeholder(tf.int32, shape=(batch_size), name='y')
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.train_offset = 0

        self.layers = [self.X]

        # Create the layers
        for layer_idx, (size_prev, size_current) in enumerate(zip([self.input_size] + layers, layers)):
            print('Creating hidden layer {}: {} -> {}'.format(layer_idx, size_prev, size_current), file=sys.stderr)
            with tf.name_scope('hidden_layer_{}'.format(layer_idx)):
                weights = tf.Variable(
                    tf.truncated_normal([size_prev, size_current],
                                        stddev=1.0 / np.sqrt(size_prev)),
                    name='weights'
                )
                biases = tf.Variable(tf.zeros([size_current]), name='biases')
                layer = tf.nn.relu(tf.matmul(self.layers[-1], weights) + biases)
                self.layers.append(layer)

        # The last layer is for the classifier
        with tf.name_scope('softmax_layer'):
            print('Creating softmax layer: {} -> {}'.format(layers[-1], self.output_size))
            weights = tf.Variable(
                tf.truncated_normal([layers[-1], self.output_size],
                                    stddev=1.0 / np.sqrt(layers[-1])),
                name='weights'
            )
            biases = tf.Variable(tf.zeros([self.output_size]), name='biases')
            self.y_logits = tf.matmul(self.layers[-1], weights) + biases

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(self.y_logits, tf.to_int64(self.y)),
            name='cross_entropy_mean_loss'
        )
        self.y_pred = tf.argmax(tf.nn.softmax(self.y_logits), 1, name='y_predictions')

        self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.y_logits, self.y, 1), tf.float32))

        self.learning_rate = tf.Variable(learning_rate, trainable=False)
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary(self.loss.op.name, self.loss)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=global_step)

        self.init = tf.initialize_all_variables()

        # Session saver and tensorboard logs
        self.logs_dir = logs_dir
        self.saver = tf.train.Saver()
        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = None

        # results
        self.results_dir = results_dir
        self.results = dict(
            train_accuracy=[],
            test_accuracy=[],
            validation_accuracy=[],
            train_precision=[],
            test_precision=[],
            validation_precision=[],
            train_recall=[],
            test_recall=[],
            validation_recall=[]
        )

    def _next_batch(self):
        start = self.train_offset
        self.train_offset += self.batch_size

        if self.train_offset > self.train_dataset.shape[0]:
            perm = np.arange(self.train_dataset.shape[0])
            np.random.shuffle(perm)
            self.train_dataset = self.train_dataset[perm]
            self.train_labels = self.train_labels[perm]
            start = 0
            self.train_offset = self.batch_size

        end = self.train_offset

        if hasattr(self.train_dataset, 'toarray'):
            return self.train_dataset[start:end].toarray(), self.train_labels[start:end]
        else:
            return self.train_dataset[start:end], self.train_labels[start:end]

    def _evaluate(self, sess, dataset, labels):
        feed_dict = {
            self.X: dataset
        }

        y_pred, accuracy = sess.run([self.y_pred, self.accuracy], feed_dict=feed_dict)

        return accuracy, precision_score(labels, y_pred), recall_score(labels, y_pred)

    def _add_results(self, dataset, accuracy, precision, recall):
        self.results['{}_accuracy'.format(dataset)].append(accuracy)
        self.results['{}_precision'.format(dataset)].append(precision)
        self.results['{}_recall'.format(dataset)].append(recall)

    def _save_results(self):
        header = ','.join(self.classes).encode('utf-8')

        # Train
        np.savetxt(os.path.join(self.results_dir, 'train_accuracy.txt'),
                   np.array(self.results['train_accuracy'], dtype=np.float32),
                   fmt='%.2f', delimiter=','.encode('utf-8'))
        np.savetxt(os.path.join(self.results_dir, 'train_precision.txt'),
                   np.array(self.results['train_precision'], dtype=np.float32),
                   fmt='%.2f', delimiter=','.encode('utf-8'),
                   header=header)
        np.savetxt(os.path.join(self.results_dir, 'train_recall.txt'),
                   np.array(self.results['train_recall'], dtype=np.float32),
                   fmt='%.2f', delimiter=','.encode('utf-8'),
                   header=header)

        # Test
        np.savetxt(os.path.join(self.results_dir, 'test_accuracy.txt'),
                   np.array(self.results['test_accuracy'], dtype=np.float32),
                   fmt='%.2f', delimiter=','.encode('utf-8'))
        np.savetxt(os.path.join(self.results_dir, 'test_precision.txt'),
                   np.array(self.results['test_precision'], dtype=np.float32),
                   fmt='%.2f', delimiter=','.encode('utf-8'),
                   header=header)
        np.savetxt(os.path.join(self.results_dir, 'test_recall.txt'),
                   np.array(self.results['test_recall'], dtype=np.float32),
                   fmt='%.2f', delimiter=','.encode('utf-8'),
                   header=header)

        # Validation
        np.savetxt(os.path.join(self.results_dir, 'validation_accuracy.txt'),
                   np.array(self.results['validation_accuracy'], dtype=np.float32),
                   fmt='%.2f', delimiter=','.encode('utf-8'))
        np.savetxt(os.path.join(self.results_dir, 'validation_precision.txt'),
                   np.array(self.results['validation_precision'], dtype=np.float32),
                   fmt='%.2f', delimiter=','.encode('utf-8'),
                   header=header)
        np.savetxt(os.path.join(self.results_dir, 'validation_recall.txt'),
                   np.array(self.results['validation_recall'], dtype=np.float32),
                   fmt='%.2f', delimiter=','.encode('utf-8'),
                   header=header)

    def train(self):
        with tf.Session() as sess:
            self.summary_writer = tf.train.SummaryWriter(self.logs_dir, sess.graph)

            sess.run(tf.initialize_all_variables())

            print('Training classifier', file=sys.stderr)
            for epoch in np.arange(self.training_epochs):
                batch_dataset, batch_labels = self._next_batch()

                feed_dict = {
                    self.X: batch_dataset,
                    self.y: batch_labels
                }

                _, loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)

                if epoch % 10 == 0:
                    print('Epoch {}: loss = {:.2f}'.format(epoch, loss), file=sys.stderr)

                    summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_str, epoch)
                    self.summary_writer.flush()

                if epoch % 100 == 0:
                    self.saver.save(sess, self.logs_dir, global_step=epoch)

                    accuracy, precision, recall = self._evaluate(sess, self.train_dataset, self.train_labels)
                    print('Training accuracy: {:.2f}'.format(accuracy), file=sys.stderr)
                    self._add_results('train', accuracy, precision, recall)

                    accuracy, precision, recall = self._evaluate(sess, self.validation_dataset, self.validation_labels)
                    print('Validation accuracy: {:.2f}'.format(accuracy), file=sys.stderr)
                    self._add_results('validation', accuracy, precision, recall)

            print('Finished training', file=sys.stderr)

            accuracy, precision, recall = self._evaluate(sess, self.test_dataset, self.test_labels)
            print('Testing accuracy: {:.2f}'.format(accuracy), file=sys.stderr)
            self._add_results('test', accuracy, precision, recall)

            print('Saving results', file=sys.stderr)
            self._save_results()
