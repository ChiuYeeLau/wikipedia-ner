# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle
import os
import sys
from scipy.io import mmwrite
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm
from wikipedianer.corpus.parser import InstanceExtractor, WikipediaCorpusColumnParser

FILES_SENTENCES = {
    "doc_01.conll": 5255631,
    "doc_02.conll": 2584456,
    "doc_03.conll": 2089800,
    "doc_04.conll": 1762920,
    "doc_05.conll": 1605431,
    "doc_06.conll": 1688569,
    "doc_07.conll": 1813089,
    "doc_08.conll": 1841952,
    "doc_09.conll": 1495951,
    "doc_10.conll": 514936
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=unicode)
    parser.add_argument("output_dir", type=unicode)
    parser.add_argument("clean_gazetteer", type=unicode)
    parser.add_argument("sloppy_gazetteer", type=unicode)

    args = parser.parse_args()
    instances = []
    labels = []

    print('Loading clean gazetteer file', file=sys.stderr)
    with open(args.clean_gazetteer, "rb") as f:
        clean_gazetteer = cPickle.load(f)

    print('Loading sloppy gazetteer file', file=sys.stderr)
    with open(args.sloppy_gazetteer, "rb") as f:
        sloppy_gazetteer = cPickle.load(f)

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
        clean_gazetteer=clean_gazetteer,
        sloppy_gazetteer=sloppy_gazetteer
    )

    for conll_file in os.listdir(args.input_dir):
        print('Getting instances from corpus {}'.format(conll_file), file=sys.stderr)

        parser = WikipediaCorpusColumnParser(os.path.join(args.input_dir, conll_file))

        for sentence in tqdm(parser, total=FILES_SENTENCES[conll_file]):
            sentence_instances, sentence_labels = instance_extractor.get_instances_for_sentence(sentence)

            instances.extend(sentence_instances)
            labels.extend(sentence_labels)

    print('Transforming features to vector', file=sys.stderr)

    vectorizer = DictVectorizer()

    X = vectorizer.fit_transform(instances)

    print('Saving matrix of features and labels', file=sys.stderr)
    mmwrite(os.path.join(args.output_dir, 'ner_feature_matrix.mtx'), X)

    with open(os.path.join(args.output_dir, 'ner_labels.pickle'), 'rb') as f:
        cPickle.dump(labels, f)

    print('All operations finished', file=sys.stderr)
