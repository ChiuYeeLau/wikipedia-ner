#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle
import gensim
import numpy as np
import os
import sys
from scipy.sparse import csr_matrix
from tqdm import tqdm
from wikipedianer.corpus.parser import InstanceExtractor, WikipediaCorpusColumnParser, WordVectorsExtractor


def parse_to_feature_matrix(input_file, output_dir, resources_dir, total_sentences):
    print('Loading resources', file=sys.stderr)

    with open(os.path.join(resources_dir, "gazetteer.pickle"), "rb") as f:
        gazetteer = cPickle.load(f)

    with open(os.path.join(resources_dir, "sloppy_gazetteer.pickle"), "rb") as f:
        sloppy_gazetteer = cPickle.load(f)

    with open(os.path.join(resources_dir, "filtered_features_names.pickle"), "rb") as f:
        features = {feature: idx for idx, feature in enumerate(cPickle.load(f))}
        features_count = len(features)

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

    instances = []
    words = []

    print('Getting instances from corpus {}'.format(input_file), file=sys.stderr)

    parser = WikipediaCorpusColumnParser(input_file)

    for sentence in tqdm(parser, total=total_sentences):
        sentence_words = [(word.idx, word.token, word.tag, word.is_doc_start) for word in sentence]
        sentence_instances, _, _ = instance_extractor.get_instances_for_sentence(sentence, 0)

        assert len(sentence_words) == len(sentence_instances)

        sentence_matrix = []
        for instance in sentence_instances:
            instance_array = np.zeros(features_count, dtype=np.float32)
            for feature, value in instance.iteritems():
                if feature in features:
                    instance_array[features[feature]] = value

            sentence_matrix.append(instance_array)

        instances.extend(sentence_matrix)
        words.extend(sentence_words)

    print('Saving matrix and words', file=sys.stderr)

    dataset_matrix = csr_matrix(np.vstack(instances))

    np.savez_compressed(os.path.join(output_dir, 'evaluation_dataset.npz'),
                        data=dataset_matrix.data, indices=dataset_matrix.indices,
                        indptr=dataset_matrix.indptr, shape=dataset_matrix.shape)

    with open(os.path.join(output_dir, 'evaluation_words.pickle'), 'wb') as f:
        cPickle.dump(words, f)


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
        sentence_words = [(word.idx, word.token, word.tag, word.is_doc_start) for word in sentence]
        sentence_instances, _, _ = instance_extractor.get_instances_for_sentence(sentence, 0)

        assert len(sentence_words) == len(sentence_instances)

        instances.extend(sentence_instances)
        words.extend(sentence_words)

    print('Saving matrix and words', file=sys.stderr)

    dataset_matrix = np.vstack(instances)

    np.savez_compressed(os.path.join(output_dir, 'evaluation_dataset_word_vectors.npz'), dataset=dataset_matrix)

    with open(os.path.join(output_dir, 'evaluation_words_word_vectors.pickle'), 'wb') as f:
        cPickle.dump(words, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file",
                        type=unicode,
                        help="Path to the text file (in column format).")
    parser.add_argument("output_dir",
                        type=unicode,
                        help="Path to store the output files")
    parser.add_argument("--resources",
                        type=unicode,
                        default=None,
                        help="Path where the resources for handcrafted features are stored")
    parser.add_argument("--wordvectors",
                        type=unicode,
                        default=None,
                        help="Path to the word vectors file")
    parser.add_argument("--total_sentences",
                        type=int,
                        default=0,
                        help="Number of sentences of the file")
    parser.add_argument("--window",
                        type=int,
                        default=2,
                        help="Size of the window vector")
    parser.add_argument("--debug",
                        action="store_true",
                        help="Debug mode")

    args = parser.parse_args()

    if args.resources is None and args.wordvectors is None:
        print('You have to give a resources path or a wordvectors path', file=sys.stderr)
        sys.exit(1)

    if args.resources is not None:
        print('Parsing to handcrafted features matrix', file=sys.stderr)
        parse_to_feature_matrix(args.input_file, args.output_dir, args.resources, args.total_sentences)

    if args.wordvectors is not None:
        print('Parsing to word vectors', file=sys.stderr)
        parse_to_word_vectors(args.input_file, args.output_dir, args.wordvectors, args.window,
                              args.total_sentences, args.debug)

    print('All operations finished', file=sys.stderr)

