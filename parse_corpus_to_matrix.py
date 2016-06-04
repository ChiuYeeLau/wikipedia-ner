# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle
import numpy as np
import os
import sys
from scipy.io import mmwrite
from scipy import sparse
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
    parser.add_argument("resources_dir", type=unicode)
    parser.add_argument("output_dir", type=unicode)
    parser.add_argument("--stopwords", action="store_true")
    parser.add_argument("--features", type=int, default=0)

    args = parser.parse_args()

    print('Loading resources', file=sys.stderr)

    with open(os.path.join(args.resources_dir, "sorted_features.pickle"), "rb") as f:
        features = cPickle.load(f)

    with open(os.path.join(args.resources_dir, "gazetteer.pickle"), "rb") as f:
        gazetteer = cPickle.load(f)

    with open(os.path.join(args.resources_dir, "sloppy_gazetteer.pickle"), "rb") as f:
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
        gazetteer=gazetteer,
        sloppy_gazetteer=sloppy_gazetteer,
        clean_gazette=True
    )

    features = {k: v for k, v in features.iteritems() if k.startswith("gazetteer") or v >= args.features}
    features_length = len(features)
    dataset_matrix = sparse.csr_matrix((0, features_length), dtype=np.int32)
    labels = []

    print('Creating vectorizer', file=sys.stderr)
    vectorizer = DictVectorizer(dtype=np.int32)
    vectorizer.feature_names_ = sorted(features.keys())
    vectorizer.vocabulary_ = {feature: idx for idx, feature in enumerate(sorted(features))}

    for conll_file in sorted(os.listdir(args.input_dir)):
        corpus_doc, _ = conll_file.split(".", 1)

        print('Getting instances from corpus {}'.format(conll_file), file=sys.stderr)

        parser = WikipediaCorpusColumnParser(os.path.join(args.input_dir, conll_file), args.stopwords)

        for sentence in tqdm(parser, total=FILES_SENTENCES[conll_file]):
            if sentence.has_named_entity:
                sentence_instances, sentence_labels = instance_extractor.get_instances_for_sentence(sentence)

                instances = vectorizer.transform(sentence_instances)

                if dataset_matrix.shape[0] == 0:
                    dataset_matrix = instances
                else:
                    dataset_matrix = sparse.vstack((dataset_matrix, instances))

                labels.extend(sentence_labels)

        if corpus_doc == "doc_04" or corpus_doc == "doc_07":
            print('Saving partial matrix', file=sys.stderr)
            mmwrite(os.path.join(args.output_dir, 'ner_feature_matrix_partial.mtx'), dataset_matrix)

            with open(os.path.join(args.output_dir, 'ner_labels_partial.pickle'), 'wb') as f:
                cPickle.dump(labels, f)
    
    print('Saving matrix of features and labels', file=sys.stderr)
    mmwrite(os.path.join(args.output_dir, 'ner_feature_matrix.mtx'), dataset_matrix)

    with open(os.path.join(args.output_dir, 'ner_labels.pickle'), 'wb') as f:
        cPickle.dump(labels, f)

    print('All operations finished', file=sys.stderr)
