# -*- coding: utf-8 -*-
import logging
import numpy

from tqdm import tqdm
from mlp import MultilayerPerceptron


class DoubleStepClassifier(object):
    """Double step classifier.
    The classification process has two stages:
        -- Use a trained classifier to determine a higher level category of an
            instance.
        -- Using the higher level category, select the corresponding low level
         classifier (entity linker) and use it to get the definitive label.

    This classifier takes three main pieces of information:
        -- The dataset and its split.
        -- The high level classifier already trained.
        -- The class of the low level classifier.
    """
    def __init__(self, x_matrix, hl_labels, ll_labels, train_indices,
                 test_indices, validation_indices, models_dirpath,
                 results_dirpath, negative_proportion=0.5,
                 low_level_classifier=MultilayerPerceptron):
        """
        :param x_matrix: a 2-dimension sparse matrix with all examples.
        :param hl_labels: an array-like object with the high level classes.
        :param ll_labels: an array-like object with the low level classes.
        :param train_indices: an array-like object with the indices of instances
            of x_matrix to use for training.
        :param test_indices: an array-like object with the indices of instances
            of x_matrix to use for testing.
        :param validation_indices: an array-like object with the indices of
            instances of x_matrix to use for validation.
        :param models_dirpath: string. The name of the directory where to store
            the trained models.
        :param results_dirpath: string. The name of the directory where to store
            the training/testing results.
        :param negative_proportion: float < 1. The relation between negative and
            positive examples to use when constructing the datasets for training
             the low level classifiers.
        """
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.validation_indices = validation_indices
        self.x_matrix = x_matrix

        self.ll_labels = ll_labels
        self.hl_labels = hl_labels

        self.models_dirpath = models_dirpath
        self.results_dirpath = results_dirpath
        self.negative_proportion = negative_proportion
        self.unique_hl_labels = numpy.unique(hl_labels)

        self.low_level_models = {}

    def create_dataset(self, indices, target_label):
        """
        Returns a numpy array with a subset of indices with balanced examples
        of target_label and negative examples, taken from labels.

        :param indices: 1-dimensional array with the indices to filter.
        :param target_label: a string, the label to consider positive.
        :return: a 1-dimensional numpy array.
        """
        if not isinstance(self.hl_labels, numpy.array):
            labels = numpy.array(self.hl_labels)
        filtered_labels = self.hl_labels[indices]
        positive_indices = indices[numpy.where(filtered_labels == target_label)]

        negative_indices = indices[numpy.where(filtered_labels != target_label)]
        # Select random subsample
        negative_indices = numpy.random.choice(negative_indices,
                                               size=positive_indices.shape[0])
        selected_indices = numpy.concatenate(positive_indices, negative_indices)
        logging.info('Selecting {} instances for label {}'.format(
            selected_indices.shape[0], target_label))
        return selected_indices

    def train(self, low_level_classifier_class, low_level_model_parameters):
        """Trains the classifier.

        :param low_level_classifier_class: python class. The class to instantiate when
            creating a low level classifier. Must extend base.BaseClassifier.
        :param low_level_model_parameters: a dictionary with the parameters to
            pass to the init function to the low level classifiers.
        """
        # Train a different model for each hl_class
        for hl_label in tqdm(self.unique_hl_labels,
                             total=self.unique_hl_labels.shape[0]):
            # Calculate indices for this high level class.
            train_indices = self.create_dataset(self.train_indices, hl_label)
            test_indices = self.create_dataset(self.test_indices, hl_label)
            validation_indices = self.create_dataset(self.validation_indices,
                                                     hl_label)

            classifier = low_level_classifier_class(
                self.x_matrix, self.ll_labels, train_indices, test_indices,
                validation_indices, **low_level_model_parameters)

    def evaluate(self, high_level_model):
        """

        :param high_level_model:
        :return:
        """
        return 0