#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
try:
    import cPickle as pickle
except ImportError:
    import pickle
import gensim
import numpy as np
import os
import sys
from scipy.sparse import coo_matrix, vstack
from tqdm import tqdm
from wikipedianer.corpus.parser import InstanceExtractor, WindowWordExtractor
from wikipedianer.corpus.parser import WikipediaCorpusColumnParser
from wikipedianer.pipeline.util import traverse_directory


def process_sentences(parser, instance_extractor, features):
    """Constructs incrementally the sparse matrix."""
    # Constructs the coo matrix incrementally.
    rows = []
    cols = []
    values = []
    labels = []
    row_count = 0  # How many rows of the matrix we have seen.

    words = []
    for sentence in tqdm(parser):
        sentence_words = [(word.idx, word.token, word.tag, word.is_doc_start)
                          for word in sentence]
        sent_instances, sent_labels, _ = instance_extractor.get_instances_for_sentence(
            sentence, 0)

        assert len(sentence_words) == len(sent_instances)
        sentence_matrix = []
        for instance in sent_instances:
            for feature, value in instance.items():
                if feature in features:
                    rows.append(row_count)
                    cols.append(features[feature])
                    values.append(value)
            row_count += 1

        labels.extend(sent_labels)
        words.extend(sentence_words)
        assert row_count == len(words)

    sentence_matrix = coo_matrix((values, (rows, cols)),
                                 shape=(row_count, len(features))).tocsr()
    return sentence_matrix, words, labels


def parse_to_feature_matrix(input_dirname, output_dir, resources_dir):
    print('Loading resources', file=sys.stderr, flush=True)

    with open(os.path.join(resources_dir, "gazetteer.p"),
              "rb") as gazetteer_file:
        gazetteer, sloppy_gazetteer = pickle.load(gazetteer_file)

    with open(os.path.join(resources_dir, "filtered_features_names.p"),
              "rb") as features_file:
        features = {feature: idx
                    for idx, feature in enumerate(pickle.load(features_file))}

    instance_extractor = InstanceExtractor(
        token=True,
        current_tag=True,
        affixes=True,
        max_ngram_length=6,
        prev_token=True,
        next_token=True,
        disjunctive_left_window=4,
        disjunctive_right_window=4,
        tag_sequence_window=2,
        gazetteer=gazetteer,
        sloppy_gazetteer=sloppy_gazetteer
    )
    print('Getting instances from corpus {}'.format(input_dirname),
          file=sys.stderr, flush=True)

    dataset_matrix = []
    words = []
    labels = []
    for file_path in sorted(traverse_directory(input_dirname, file_pattern='*.conll')):
        parser = WikipediaCorpusColumnParser(file_path)
        file_matrix, file_words, file_labels = process_sentences(
            parser, instance_extractor, features)
        dataset_matrix.append(file_matrix)
        words.extend(file_words)
        labels.extend(file_labels)
    dataset_matrix = vstack(dataset_matrix)

    print('Saving matrix and words', file=sys.stderr, flush=True)
    np.savez_compressed(
        os.path.join(output_dir, 'evaluation_dataset.npz'),
        data=dataset_matrix.data, indices=dataset_matrix.indices,
        indptr=dataset_matrix.indptr, shape=dataset_matrix.shape)

    with open(os.path.join(output_dir, 'evaluation_words.pickle'),
              'wb') as output_file:
        pickle.dump(words, output_file)

    with open(os.path.join(output_dir, 'labels.pickle'), 'wb') as output_file:
        pickle.dump(labels, output_file)


def parse_to_word_windows(input_dirname, output_dir, window):
    instance_extractor = WindowWordExtractor(window) 

    instances = []
    words = []
    labels = []

    print('Getting instances from corpus {}'.format(input_dirname), \
          file=sys.stderr, flush=True)

    for file_path in sorted(traverse_directory(
            input_dirname, file_pattern='*.conll')):
        parser = WikipediaCorpusColumnParser(file_path)
        for sentence in tqdm(parser):
            sentence_words = [
                (word.idx, word.token, word.tag, word.is_doc_start)
                for word in sentence]
            sentence_instances, sentence_labels, _ = \
                instance_extractor.get_instances_for_sentence(sentence, 0)

            assert len(sentence_words) == len(sentence_instances)

            labels.extend(sentence_labels)
            instances.extend(sentence_instances)
            words.extend(sentence_words)

    print('Saving matrix and words', file=sys.stderr, flush=True)
    instances_file = os.path.join(output_dir, 'evaluation_dataset_word_vectors.pickle')
    with open(instances_file, 'wb') as outfile:
        pickle.dump(instances, outfile) 

    with open(os.path.join(output_dir, 'evaluation_words_word_vectors.pickle'), 'wb') as f:
        pickle.dump(words, f)

    with open(os.path.join(output_dir, 'labels.pickle'), 'wb') as f:
        pickle.dump(labels, f)


def parse_arguments():
    """Returns the stdin arguments"""
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("input_dirname", type=str,
                           help="Path to the directory with the text files "
                                "(in column format).")
    arg_parse.add_argument("output_dir", type=str,
                           help="Path to store the output files")
    arg_parse.add_argument("--resources", type=str, default=None,
                           help="Path where the resources for handcrafted "
                                "features are stored")
    arg_parse.add_argument("--window", type=int, default=2,
                           help="Size of the window vector")
    return arg_parse.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.resources is not None:
        print('Parsing to handcrafted features matrix', file=sys.stderr, flush=True)
        parse_to_feature_matrix(args.input_dirname, args.output_dir,
                                args.resources)
    else:
        print('Parsing to word vectors', file=sys.stderr, flush=True)
        parse_to_word_windows(args.input_dirname, args.output_dir,
                              args.window)

    print('All operations finished', file=sys.stderr, flush=True)

