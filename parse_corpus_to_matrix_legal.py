# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle
import numpy as np
import os
import sys
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm
from wikipedianer.corpus.parser import InstanceExtractor, WikipediaCorpusColumnParser

FILES_SENTENCES = {
    "doc_01.conll": 2962737,
    "doc_02.conll": 3156576,
    "doc_03.conll": 2574401,
    "doc_04.conll": 2379707,
    "doc_05.conll": 2495369,
    "doc_06.conll": 2493490,
    "doc_07.conll": 475036,
    "doc_08.conll": 2994167,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=unicode)
    parser.add_argument("resources_dir", type=unicode)
    parser.add_argument("output_dir", type=unicode)
    parser.add_argument("--stopwords", action="store_true")

    args = parser.parse_args()

    print('Loading resources', file=sys.stderr)

    with open(os.path.join(args.resources_dir, "gazetteer.pickle"), "rb") as f:
        gazetteer = cPickle.load(f)

    with open(os.path.join(args.resources_dir, "sloppy_gazetteer.pickle"), "rb") as f:
        sloppy_gazetteer = cPickle.load(f)

    try:
        valid_indices = np.load(os.path.join(args.resources_dir, "valid_indices.npz"))
    except IOError:
        valid_indices = {'nne_instances': [], 'ne_instances': []}

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
        valid_indices=set(valid_indices['nne_instances']).union(set(valid_indices['ne_instances']))
    )

    instances = []
    labels = []
    word_index = 0

    for conll_file in sorted(os.listdir(args.input_dir)):
        corpus_doc, _ = conll_file.split(".", 1)

        print('Getting instances from corpus {}'.format(conll_file), file=sys.stderr)

        parser = WikipediaCorpusColumnParser(os.path.join(args.input_dir, conll_file), args.stopwords)

        for sentence in tqdm(parser, total=FILES_SENTENCES[conll_file]):
            if sentence.has_named_entity:
                sentence_instances, sentence_labels, word_index = \
                    instance_extractor.get_instances_for_sentence(sentence, word_index)

                instances.extend(sentence_instances)
                labels.extend(sentence_labels)

        if corpus_doc == "doc_03" or corpus_doc == "doc_06":
            print('Saving partial matrix', file=sys.stderr)

            vectorizer = DictVectorizer(dtype=np.int32)

            dataset_matrix = vectorizer.fit_transform(instances)

            del vectorizer

            np.savez_compressed(os.path.join(args.output_dir, 'ner_feature_matrix_partial.npz'),
                                data=dataset_matrix.data, indices=dataset_matrix.indices,
                                indptr=dataset_matrix.indptr, shape=dataset_matrix.shape)

            del dataset_matrix

            with open(os.path.join(args.output_dir, 'ner_labels_partial.pickle'), 'wb') as f:
                cPickle.dump(labels, f)

    print('Saving matrix of features and labels', file=sys.stderr)

    vectorizer = DictVectorizer(dtype=np.int32)

    dataset_matrix = vectorizer.fit_transform(instances)

    del vectorizer

    np.savez_compressed(os.path.join(args.output_dir, 'ner_feature_matrix.npz'),
                        data=dataset_matrix.data, indices=dataset_matrix.indices,
                        indptr=dataset_matrix.indptr, shape=dataset_matrix.shape)

    del dataset_matrix

    with open(os.path.join(args.output_dir, 'ner_labels.pickle'), 'wb') as f:
        cPickle.dump(labels, f)

    print('All operations finished', file=sys.stderr)
