# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle as pickle
import numpy as np
import os
import re
import sys

from collections import Counter
from wikipedianer.dataset.preprocess import StratifiedSplitter


def ner_label_replace(labels, mappings):
    for label in labels:
        if label.startswith('O'):
            yield 'O'
        else:
            yield 'I'


def ne_person_label_replace(labels, mappings):
    for label in labels:
        label = re.sub(r'^[BI]-', '', label)
        if 'no_person' in mappings.get(label, set()):
            yield 'no_person'
        elif 'wordnet_person_100007846' in mappings.get(label, set()):
            yield 'wordnet_person_100007846'
        else:
            yield 'O'


def ne_category_label_replace(labels, mappings):
    for label in labels:
        label = re.sub(r'^[BI]-', '', label)
        if 'wordnet_movie_106613686' in mappings.get(label, set()):
            yield 'wordnet_movie_106613686'
        elif 'wordnet_soundtrack_104262969' in mappings.get(label, set()):
            yield 'wordnet_soundtrack_104262969'
        elif 'wordnet_actor_109765278' in mappings.get(label, set()):
            yield 'wordnet_actor_109765278'
        elif 'wordnet_film_director_110088200' in mappings.get(label, set()):
            yield 'wordnet_film_director_110088200'
        elif 'wordnet_film_maker_110088390' in mappings.get(label, set()):
            yield 'wordnet_film_maker_110088390'
        else:
            yield 'O'


def ne_uri_label_replace(labels, mappings):
    for label in labels:
        yield re.sub(r"^B-", "I-", label)


labels_replacements = [
    ("NER", ner_label_replace),
    ("NEP", ne_person_label_replace),
    ("NEC", ne_category_label_replace),
    ("NEU", ne_uri_label_replace)
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path", type=unicode)
    parser.add_argument("save_path", type=unicode)
    parser.add_argument("mappings", type=unicode)
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--validation_size", type=float, default=0.1)
    parser.add_argument("--min_count", type=int, default=50)

    args = parser.parse_args()

    print('Loading labels from file {}'.format(args.labels_path), file=sys.stderr)
    with open(args.labels_path, 'rb') as f:
        labels = np.array(pickle.load(f))

    with open(args.mappings, 'rb') as f:
        class_mappings = pickle.load(f)

    for category_name, replacement_function in labels_replacements:
        print('Getting replaced labels for category {}'.format(category_name), file=sys.stderr)
        replaced_labels = list(replacement_function(labels, class_mappings))

        print('Getting filtered classes for category {}'.format(category_name), file=sys.stderr)
        filtered_classes = {l for l, v in Counter(replaced_labels).iteritems() if v >= args.min_count}

        print('Getting filtered indices for category {}'.format(category_name), file=sys.stderr)
        filtered_indices = np.array([i for i, l in enumerate(replaced_labels) if l in filtered_classes],
                                    dtype=np.int32)

        strat_split = StratifiedSplitter(np.array(replaced_labels), filtered_indices)

        print('Splitting the dataset', file=sys.stderr)
        strat_split.split_dataset(train_size=args.train_size, test_size=args.test_size,
                                  validation_size=args.validation_size)

        print('Saving splitted indices', file=sys.stderr)
        strat_split.save_splitted_dataset_indices(os.path.join(args.save_path, "{}_indices.npz".format(category_name)))
