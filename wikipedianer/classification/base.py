# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals
import pandas

class BaseClassifier(object):
    def __init__(self):
        self.test_results = pandas.DataFrame(
            columns=['accuracy', 'class', 'precision', 'recall', 'fscore'])

    def add_test_results(self, accuracy, precision, recall, fscore, classes):
        self.test_results = self.test_results.append({'accuracy': accuracy},
                                                     ignore_index=True)
        for cls_idx, cls in enumerate(classes):
            self.test_results = self.test_results.append({
                'class': cls,
                'precision': precision[cls_idx],
                'recall': recall[cls_idx],
                'fscore': fscore[cls_idx]
            }, ignore_index=True)
