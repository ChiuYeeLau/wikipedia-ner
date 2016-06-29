# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle
import numpy as np
import os
import sys
from scipy.sparse import csr_matrix
from tqdm import tqdm
from wikipedianer.corpus.parser import InstanceExtractor, WikipediaCorpusColumnParser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=unicode)
    parser.add_argument("resources_dir", type=unicode)
    parser.add_argument("output_dir", type=unicode)
    parser.add_argument("--stopwords", action="store_true")
    parser.add_argument("--total_sentences", type=int, default=2898)

    args = parser.parse_args()

    print('Loading resources', file=sys.stderr)

    with open(os.path.join(args.resources_dir, "gazetteer.pickle"), "rb") as f:
        gazetteer = cPickle.load(f)

    with open(os.path.join(args.resources_dir, "sloppy_gazetteer.pickle"), "rb") as f:
        sloppy_gazetteer = cPickle.load(f)

    with open(os.path.join(args.resources_dir, "filtered_features_names.pickle"), "rb") as f:
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

    print('Getting instances from corpus {}'.format(args.input_file), file=sys.stderr)

    parser = WikipediaCorpusColumnParser(args.input_file, args.stopwords)

    for sentence in tqdm(parser, total=args.total_sentences):
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

    np.savez_compressed(os.path.join(args.output_dir, 'evaluation_dataset.npz'),
                        data=dataset_matrix.data, indices=dataset_matrix.indices,
                        indptr=dataset_matrix.indptr, shape=dataset_matrix.shape)

    with open(os.path.join(args.output_dir, 'evaluation_words.pickle'), 'wb') as f:
        cPickle.dump(words, f)

    print('All operations finished', file=sys.stderr)
