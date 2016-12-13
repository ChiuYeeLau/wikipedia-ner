"""LSTM Keras classifier."""

import numpy
import tensorflow as tf


from base import BaseClassifier
from tensorflow.python.ops import rnn, rnn_cell


class LSTMClassifier(object):
    """LSTM tensorflow classifier."""
    def __init__(self):
        """Creates a LSTM classifier.

        Args:
            input_dim: size of each element. (Number of features)
            input_length: size of each sequence.
            lstm_state_size:
        """
        # Build the network
        self.learning_rate = 0.001
        self.training_iters = 100
        self.batch_size = 128
        self.display_step = 10

        self.session = None

    def fit(self, train_x, train_y):
        """Trains the classifier"""
        n_input = train_x.shape[-1]  # Number of features in each instance
        n_steps = train_x.shape[1]  # Number of instances in each sequence
        n_hidden = 128  # hidden layer num of features
        n_classes = train_y.shape[1]  # Total number of classes

        # tf Graph input
        self.x_placeholder = tf.placeholder("float", [None, n_steps, n_input])
        self.y_placeholder = tf.placeholder("float", [None, n_classes])

        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        def RNN(x, weights, biases):
            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, n_steps, n_input)
            # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

            # Permuting batch_size and n_steps
            x = tf.transpose(x, [1, 0, 2])
            # Reshaping to (n_steps*batch_size, n_input)
            x = tf.reshape(x, [-1, n_input])
            # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
            x = tf.split(0, n_steps, x)

            # Define a lstm cell with tensorflow
            lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

            # Get lstm cell output
            outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

            # Linear activation, using rnn inner loop last output
            return tf.matmul(outputs[-1], weights['out']) + biases['out']

        self.prediction = RNN(self.x_placeholder, weights, biases)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            self.prediction, self.y_placeholder))
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y_placeholder, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initializing the variables
        init = tf.initialize_all_variables()

        self.session = tf.Session()
        self.session.run(init)
        # Keep training until reach max iterations
        for iter, batch_start in enumerate(
                range(0, self.training_iters, self.batch_size)):
            batch_x = train_x[batch_start:batch_start+self.batch_size]
            batch_y = train_y[batch_start:batch_start+self.batch_size]
            # Reshape data to get 28 seq of 28 elements
            # batch_x = batch_x.reshape((self.batch_size, n_steps, n_input))
            # Run optimization op (backprop)
            self.session.run(optimizer, feed_dict={
                self.x_placeholder: batch_x,
                self.y_placeholder: batch_y})
            if iter % self.display_step == 0:
                # Calculate batch accuracy
                acc = self.session.run(self.accuracy, feed_dict={
                    self.x_placeholder: batch_x,
                    self.y_placeholder: batch_y
                })
                # Calculate batch loss
                loss = self.session.run(cost, feed_dict={
                    self.x_placeholder: batch_x,
                    self.y_placeholder: batch_y})
                print ("Iter {}, Minibatch Loss={:.6f}, Training Accuracy= "
                       "{:.5f}").format(iter, loss, acc)
        print "Optimization Finished!"

    def predict(self, train_x):
        return self.prediction.eval(
            session=self.session, feed_dict={self.x_placeholder: train_x})

    def __del__(self):
        self.session.close()


class SequentialPipeline(BaseClassifier):
    """Pipeline for sequential classifier for distributed datasets."""
    def __init__(self, dataset, labels, train_indices, test_indices,
                 validation_indices):
        super(SequentialPipeline, self).__init__(dataset, labels, train_indices,
                                             test_indices, validation_indices)
        self.classifier = LSTMClassifier()

    def train(self, sequence_max_length):
        # Pad missing sequences with zero
        self.pad_sequences(self.train_dataset, sequence_max_length)
        self.classifier.fit(self.train_dataset, self.train_labels)

    @staticmethod
    def pad_sequences(x_matrix, sequence_max_length=10):
        # Get the number of dimensions
        if len(x_matrix[0].shape) == 2:
            pad_width = lambda shape: [(0, sequence_max_length - shape), (0, 0)]
        else:
            pad_width = lambda shape: (0, sequence_max_length - shape)
        for index, sequence in enumerate(x_matrix):
            if sequence.shape[0] > sequence_max_length:
                x_matrix[index] = sequence[:sequence_max_length]
            elif sequence.shape[0] < sequence_max_length:
                sequence = numpy.pad(sequence, pad_width(sequence.shape[0]),
                                     mode='constant')
                x_matrix[index] = sequence
        return numpy.array(x_matrix)