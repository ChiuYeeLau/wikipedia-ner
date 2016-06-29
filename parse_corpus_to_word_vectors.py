# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle
import gensim
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import SENTENCES
from wikipedianer.corpus.parser import WordVectorsExtractor, WikipediaCorpusColumnParser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=unicode)
    parser.add_argument("wordvectors", type=unicode)
    parser.add_argument("output_dir", type=unicode)
    parser.add_argument("--valid_indices", type=unicode, default='')
    parser.add_argument("--experiment_kind", type=unicode, default="legal")
    parser.add_argument("--stopwords", action="store_true")
    parser.add_argument("--window", type=int, default=5)

    args = parser.parse_args()

    print('Loading vectors', file=sys.stderr)
    word2vec_model = gensim.models.Word2Vec.load_word2vec_format(args.wordvectors, binary=True)

    try:
        valid_indices = np.load(os.path.join(args.resources_dir, "valid_indices.npz"))
    except IOError:
        valid_indices = {'nne_instances': [], 'ne_instances': []}

    instance_extractor = WordVectorsExtractor(
        word2vec_model, args.window, set(valid_indices['nne_instances']).union(set(valid_indices['ne_instances']))
    )

    instances = []
    labels = []
    word_index = 0

    for conll_file in sorted(os.listdir(args.input_dir)):
        corpus_doc, _ = conll_file.split(".", 1)

        print('Getting instances from corpus {}'.format(conll_file), file=sys.stderr)

        parser = WikipediaCorpusColumnParser(os.path.join(args.input_dir, conll_file), args.stopwords)

        for sentence in tqdm(parser, total=SENTENCES[args.experiment_kind][conll_file]):
            if sentence.has_named_entity:
                sentence_instances, sentence_labels, word_index = \
                    instance_extractor.get_instances_for_sentence(sentence, word_index)

                instances.extend(sentence_instances)
                labels.extend(sentence_labels)

    print('Saving matrix of features and labels', file=sys.stderr)
    np.savez_compressed(os.path.join(args.output_dir, 'ner_word_vectors_matrix.npz'), dataset=np.vstack(instances))
    with open(os.path.join(args.output_dir, 'ner_word_vectors_labels.pickle'), 'wb') as f:
        cPickle.dump(labels, f)

    print('All operations finished', file=sys.stderr)
