"""Logistic Regression wrapper over sklearn library for compatibility with
DoubleStepClassifier."""

from __future__ import absolute_import, print_function, unicode_literals

import os
import numpy
import pandas
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from .base import BaseClassifier
from .double_step_classifier import ClassifierFactory


class LRClassifierFactory(ClassifierFactory):

    def __init__(self, save_models=False, results_save_path=''):
        self.save_models = save_models
        self.results_save_path = results_save_path

    def get_classifier(self, dataset, experiment_name):
        return LRCLassifier(dataset, save_model=self.save_models,
                            results_save_path=self.results_save_path,
                            experiment_name=experiment_name)


class LRCLassifier(BaseClassifier):

    def __init__(self, dataset, cl_iteration=1, save_model=False,
                 results_save_path=None, experiment_name=''):
        super(LRCLassifier, self).__init__()
        self.model = LogisticRegression()
        self.dataset = dataset
        self.cl_iteration = cl_iteration
        self.save_model = save_model
        self.results_save_path = results_save_path
        self.experiment_name = experiment_name

    def train(self, *args, **kwargs):
        # Train
        self.model.fit(
            self.dataset.datasets['train'].data,
            self.dataset.datasets['train'].labels[:,self.cl_iteration])

        # Get accuracy on test dataset
        accuracy, precision, recall, fscore, y_true, y_pred = self.evaluate(
            'test', return_extras=True)
        predictions_results = pandas.DataFrame(numpy.vstack([y_true, y_pred]).T,
                                                    columns=['true', 'prediction'])
        predictions_results.to_csv(self.get_predictions_filename(), index=False)
        self.add_test_results(accuracy, precision, recall, fscore,
                              self.dataset.classes[1])
        print(self.test_results)

        if self.save_model:
            self.save()

    def evaluate(self, dataset_name='test', return_extras=False, restore=False,
                 *args, **kwargs):
        if restore:
            self.read()
        y_true = self.dataset.datasets[dataset_name].labels[:,self.cl_iteration]
        y_pred = self.model.predict(self.dataset.datasets[dataset_name].data)
        accuracy = accuracy_score(y_true, y_pred.astype(y_true.dtype))
        if not return_extras:
            return accuracy
        else:
            labels = numpy.arange(
                self.dataset.output_size(self.cl_iteration))
            precision = precision_score(y_true, y_pred, labels=labels,
                                        average=None)
            recall = recall_score(y_true, y_pred, labels=labels,
                                  average=None)
            fscore = f1_score(y_true, y_pred, labels=labels,
                              average=None)

            return accuracy, precision, recall, fscore, y_true, y_pred

    def get_save_filename(self):
        return os.path.join(self.results_save_path,
                            '{}_lr.model'.format(self.experiment_name))

    def get_predictions_filename(self):
        return os.path.join(
            self.results_save_path,
            'test_predictions_{}.csv'.format(self.experiment_name))

    def save(self):
        with open(self.get_save_filename(), 'wb') as out_file:
            pickle.dump(self.model, out_file)

    def read(self):
        with open(self.get_save_filename(), 'rb') as in_file:
            self.model = pickle.load(in_file)
