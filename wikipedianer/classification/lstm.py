"""LSTM Keras classifier."""

import numpy

from base import BaseClassifier
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.regularizers import l2


class LSTMClassifier(object):
    """LSTM Keras classifier."""
    def __init__(self, input_dim, lstm_state_size=100):
        # Build the network
        self.model = Sequential()
        self.model.add(LSTM(
            lstm_state_size, input_dim=input_dim, W_regularizer=l2(0.01),
            U_regularizer=l2(0.01), b_regularizer=l2(0.01),
            dropout_W=0.1, dropout_U=0.1, return_sequences=True))
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])

    def fit(self, train_x, train_y):
        """Trains the classifier"""
        self.model.fit(train_x, train_y, nb_epoch=100, batch_size=1, verbose=2)

    def predict(self, train_x):
        self.model.predict(train_x)


class SequentialPipeline(BaseClassifier):
    """Pipeline for sequential classifier for distributed datasets."""
    def __init__(self, dataset, labels, train_indices, test_indices,
                 validation_indices):
        super(LSTMClassifier, self).__init__(dataset, labels, train_indices,
                                             test_indices, validation_indices)
        self.classifier = LSTMClassifier(self.train_dataset.shape[1])

    def train(self):
        self.classifier.fit(self.train_dataset, self.train_labels)