#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import SENTENCES
from wikipedianer.corpus.parser import WikipediaCorpusColumnParser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=unicode)
    parser.add_argument("output_file", type=unicode)
    parser.add_argument("--sentences", type=unicode, default="movies")
    parser.add_argument("--stopwords", action="store_true")

    args = parser.parse_args()

    labels = []

    for conll_file in sorted(filter(lambda f: f.endswith(".conll"), os.listdir(args.input_dir))):
        corpus_doc, _ = conll_file.split(".", 1)

        print('Parsing {}'.format(corpus_doc), file=sys.stderr)

        parser = WikipediaCorpusColumnParser(os.path.join(args.input_dir, conll_file), args.stopwords)

        for sentence in tqdm(parser, total=SENTENCES[args.sentences][conll_file]):
            if sentence.has_named_entity:
                labels.extend(sentence.io_labels)

    print('Counting labels', file=sys.stderr)
    unique_labels, inverse_indices, counts = np.unique(labels, return_inverse=True, return_counts=True)
    counts.sort()
    subsample_count = min(counts[:-1].sum(), counts[-1])
    nne_index = np.where(unique_labels == 'O')[0][0]
    nne_instances = np.random.permutation(np.where(inverse_indices == nne_index)[0])[:subsample_count]
    ne_instances = np.where(inverse_indices != nne_index)[0]

    print('Saving indices to {}'.format(args.output_file), file=sys.stderr)
    np.savez_compressed(args.output_file, nne_instances=nne_instances, ne_instances=ne_instances)

    print('Finished getting the data', file=sys.stderr)
