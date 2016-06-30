# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle
import gensim
import numpy as np
import os
import sys
from tqdm import tqdm
from wikipedianer.corpus.parser import WikipediaCorpusColumnParser, WordVectorsExtractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=unicode)
    parser.add_argument("wordvectors", type=unicode)
    parser.add_argument("output_dir", type=unicode)
    parser.add_argument("--stopwords", action="store_true")
    parser.add_argument("--total_sentences", type=int, default=2898)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    print('Loading resources', file=sys.stderr)

    print('Loading vectors', file=sys.stderr)
    if not args.debug:
        word2vec_model = gensim.models.Word2Vec.load_word2vec_format(args.wordvectors, binary=True)
    else:
        word2vec_model = gensim.models.Word2Vec()

    instance_extractor = WordVectorsExtractor(word2vec_model, args.window)

    instances = []
    words = []

    print('Getting instances from corpus {}'.format(args.input_file), file=sys.stderr)

    parser = WikipediaCorpusColumnParser(args.input_file, args.stopwords)

    for sentence in tqdm(parser, total=args.total_sentences):
        sentence_words = [(word.idx, word.token, word.tag, word.is_doc_start) for word in sentence]
        sentence_instances, _, _ = instance_extractor.get_instances_for_sentence(sentence, 0)

        assert len(sentence_words) == len(sentence_instances)

        instances.extend(sentence_instances)
        words.extend(sentence_words)

    print('Saving matrix and words', file=sys.stderr)

    dataset_matrix = np.vstack(instances)

    np.savez_compressed(os.path.join(args.output_dir, 'evaluation_dataset_word_vectors.npz'), datataset=dataset_matrix)

    with open(os.path.join(args.output_dir, 'evaluation_words_word_vectors.pickle'), 'wb') as f:
        cPickle.dump(words, f)

    print('All operations finished', file=sys.stderr)
