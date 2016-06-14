# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle
import gensim
import numpy as np
import os
import sys
from scipy import sparse
from tqdm import tqdm
from wikipedianer.corpus.parser import WordVectorsExtractor, WikipediaCorpusColumnParser

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
    parser.add_argument("wordvectors", type=unicode)
    parser.add_argument("output_dir", type=unicode)
    parser.add_argument("--stopwords", action="store_true")
    parser.add_argument("--window", type=int, default=2)

    args = parser.parse_args()

    print('Loading vectors', file=sys.stderr)
    word2vec_model = gensim.models.Word2Vec.load_word2vec_format(args.wordvectors, binary=True)

    instance_extractor = WordVectorsExtractor(word2vec_model, args.window)
    dataset_matrix = []
    labels = []

    for conll_file in sorted(os.listdir(args.input_dir)):
        corpus_doc, _ = conll_file.split(".", 1)

        print('Getting instances from corpus {}'.format(conll_file), file=sys.stderr)

        parser = WikipediaCorpusColumnParser(os.path.join(args.input_dir, conll_file), args.stopwords)

        for sentence in tqdm(parser, total=FILES_SENTENCES[conll_file]):
            if sentence.has_named_entity:
                sentence_instances, sentence_labels = instance_extractor.get_instances_for_sentence(sentence)

                dataset_matrix.extend(sentence_instances)
                labels.extend(sentence_labels)

        if corpus_doc != "doc_01":
            print('Loading partial matrix and labels', file=sys.stderr)
            partial_matrix = np.load(os.path.join(args.output_dir, 'ner_word_vectors_matrix.npz'))['dataset']

            with open(os.path.join(args.output_dir, 'ner_word_vectors_labels.pickle'), 'rb') as f:
                partial_labels = cPickle.load(f)

            partial_matrix = np.vstack((partial_matrix, np.vstack(dataset_matrix)))
            partial_labels.extend(labels)
        else:
            partial_matrix = np.vstack(dataset_matrix)
            partial_labels = labels

        print('Saving partial matrix', file=sys.stderr)
        np.savez_compressed(os.path.join(args.output_dir, 'ner_word_vectors_matrix.npz'),
                            dataset=partial_matrix)

        with open(os.path.join(args.output_dir, 'ner_word_vectors_labels.pickle'), 'wb') as f:
            cPickle.dump(partial_labels, f)
    
    print('All operations finished', file=sys.stderr)
