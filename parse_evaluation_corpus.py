#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle
import csv
import gensim
import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys

from scipy.sparse import coo_matrix
from tqdm import tqdm
from wikipedianer.corpus.parser import InstanceExtractor, WordVectorsExtractor
from wikipedianer.corpus.parser import WikipediaCorpusColumnParser


if sys.version_info.major == 3:
    unicode = str


def process_sentences(parser, total_sentences, instance_extractor, features):
    """Constructs incrementally the sparse matrix."""
    # Constructs the coo matrix incrementally.
    rows = []
    cols = []
    values = []
    row_count = 0  # How many rows of the matrix we have seen.

    words = []
    for sentence in tqdm(parser, total=total_sentences):
        sentence_words = [(word.idx, word.token.encode('utf-8'), word.tag, word.is_doc_start)
                          for word in sentence]
        sent_instances, _, _ = instance_extractor.get_instances_for_sentence(
            sentence, 0)

        assert len(sentence_words) == len(sent_instances)
        for instance in sent_instances:
            for feature, value in instance.items():
                if feature in features:
                    rows.append(row_count)
                    cols.append(features[feature])
                    values.append(value)
            row_count += 1

        words.extend(sentence_words)
        assert row_count == len(words)

    sentence_matrix = coo_matrix((values, (rows, cols)),
                                 shape=(row_count, len(features))).tocsr()
    return sentence_matrix, words


def parse_to_feature_matrix(input_file, output_dir, resources_dir,
                            total_sentences):
    print('Loading resources', file=sys.stderr)

    with open(os.path.join(resources_dir, "gazetteer.pickle"),
              "rb") as gazetteer_file:
        gazetteer = pickle.load(gazetteer_file)

    try:
        with open(os.path.join(resources_dir, "sloppy_gazetteer.pickle"),
                  "rb") as sloppy_gazetteer_file:
            sloppy_gazetteer = pickle.load(sloppy_gazetteer_file)
    except FileNotFoundError:
        sloppy_gazetteer = set()

    with open(os.path.join(resources_dir, "filtered_features_names.pickle"),
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
    print('Getting instances from corpus {}'.format(input_file),
          file=sys.stderr)

    parser = WikipediaCorpusColumnParser(input_file)
    dataset_matrix, words = process_sentences(parser, total_sentences,
                                              instance_extractor, features)

    print('Saving matrix and words', file=sys.stderr)
    np.savez_compressed(
        os.path.join(output_dir, 'evaluation_dataset.npz'),
        data=dataset_matrix.data, indices=dataset_matrix.indices,
        indptr=dataset_matrix.indptr, shape=dataset_matrix.shape)

    with open(os.path.join(output_dir, 'evaluation_words.csv'),
              'wb') as output_file:
        pickle.dump(words, output_file)


def parse_to_word_vectors(input_file, output_dir, wordvectors, window, total_sentences, debug):
    print('Loading vectors', file=sys.stderr)
    if not debug:
        word2vec_model = gensim.models.Word2Vec.load_word2vec_format(wordvectors, binary=True)
    else:
        word2vec_model = gensim.models.Word2Vec()

    instance_extractor = WordVectorsExtractor(word2vec_model, window)

    instances = []
    words = []

    print('Getting instances from corpus {}'.format(input_file), file=sys.stderr)

    parser = WikipediaCorpusColumnParser(input_file)

    for sentence in tqdm(parser, total=total_sentences):
        sentence_words = [(word.idx, word.token, word.tag, word.is_doc_start)
                          for word in sentence]
        sentence_instances, _, _ = instance_extractor.get_instances_for_sentence(sentence, 0)

        assert len(sentence_words) == len(sentence_instances)

        instances.extend(sentence_instances)
        words.extend(sentence_words)

    print('Saving matrix and words', file=sys.stderr)

    dataset_matrix = np.vstack(instances)

    np.savez_compressed(os.path.join(output_dir, 'evaluation_dataset_word_vectors.npz'), dataset=dataset_matrix)

    with open(os.path.join(output_dir, 'evaluation_words_word_vectors.pickle'), 'wb') as f:
        pickle.dump(words, f)


def parse_arguments():
    """Returns the stdin arguments"""
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("input_file", type=str,
                           help="Path to the text file (in column format).")
    arg_parse.add_argument("output_dir", type=str,
                           help="Path to store the output files")
    arg_parse.add_argument("--resources", type=str, default=None,
                           help="Path where the resources for handcrafted "
                                "features are stored")
    arg_parse.add_argument("--wordvectors", type=str, default=None,
                           help="Path to the word vectors file")
    arg_parse.add_argument("--total_sentences", type=int, default=0,
                           help="Number of sentences of the file")
    arg_parse.add_argument("--window", type=int, default=2,
                           help="Size of the window vector")
    arg_parse.add_argument("--debug", action="store_true", help="Debug mode")
    return arg_parse.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.resources is None and args.wordvectors is None:
        print('You have to give a resources path or a wordvectors path',
              file=sys.stderr)
        sys.exit(1)

    if args.resources is not None:
        print('Parsing to handcrafted features matrix', file=sys.stderr)
        parse_to_feature_matrix(args.input_file, args.output_dir,
                                args.resources, args.total_sentences)

    if args.wordvectors is not None:
        print('Parsing to word vectors', file=sys.stderr)
        parse_to_word_vectors(args.input_file, args.output_dir,
                              args.wordvectors, args.window,
                              args.total_sentences, args.debug)

    print('All operations finished', file=sys.stderr)

