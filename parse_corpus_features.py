#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle
import os
import sys
from tqdm import tqdm
from wikipedianer.corpus.parser import FeatureExtractor, WikipediaCorpusColumnParser

FILES_SENTENCES = {
    "doc_01": 5255631,
    "doc_02": 2584456,
    "doc_03": 2089800,
    "doc_04": 1762920,
    "doc_05": 1605431,
    "doc_06": 1688569,
    "doc_07": 1813089,
    "doc_08": 1841952,
    "doc_09": 1495951,
    "doc_10": 514936
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=unicode)
    parser.add_argument("output_dir", type=unicode)
    parser.add_argument("--stopwords", action="store_true")

    args = parser.parse_args()

    feature_extractor = FeatureExtractor(
        max_ngram_length=6,
        disjunctive_left_window=4,
        disjunctive_right_window=4,
        tag_sequence_window=2
    )
    gazetteer = set()

    for conll_file in sorted(os.listdir(args.input_dir)):
        corpus_doc, _ = conll_file.split(".", 1)

        print('Collecting features for corpus {}'.format(corpus_doc), file=sys.stderr)

        parser = WikipediaCorpusColumnParser(os.path.join(args.input_dir, conll_file), args.stopwords)

        for sentence in tqdm(parser, total=FILES_SENTENCES[corpus_doc]):
            if sentence.has_named_entity:
                feature_extractor.features_from_sentence(sentence)
                gazetteer.update(sentence.get_gazettes())

    print('Updating features with gazette features', file=sys.stderr)
    feature_extractor.update_features({'gazette:{}'.format(gazette for gazette in gazetteer)})

    print('Saving sorted features', file=sys.stderr)
    feature_extractor.save_sorted_features(os.path.join(args.output_dir, 'sorted_features.pickle'))

    print('Saving gazetteer', file=sys.stderr)
    with open(os.path.join(args.output_dir, 'gazetteer.pickle', 'wb')) as f:
        cPickle.dump(gazetteer, f)

    print('All operations finished', file=sys.stderr)
