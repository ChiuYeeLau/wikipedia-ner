# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle
import os
import sys
from collections import defaultdict
from tqdm import tqdm
from utils import SENTENCES
from wikipedianer.corpus.parser import WikipediaCorpusColumnParser


def collect_gazeteers(input_dir, output_dir, experiment_kind):
    if experiment_kind not in SENTENCES:
        raise Exception('Not a valid experiment (must be "legal" or "movies")')

    gazetteer = defaultdict(int)

    for corpus_file in sorted(os.listdir(input_dir)):
        corpus_doc, _ = corpus_file.split(".", 1)
        print('Getting gazettes for corpus {}'.format(corpus_doc), file=sys.stderr)

        parser = WikipediaCorpusColumnParser(os.path.join(input_dir, corpus_file), remove_stop_words=True)

        for sentence in tqdm(parser, total=SENTENCES[experiment_kind][corpus_file]):
            if sentence.has_named_entity:
                for gazette, value in sentence.get_gazettes().iteritems():
                    gazetteer[gazette] += value

    print('Saving gazetteer', file=sys.stderr)
    with open(os.path.join(output_dir, 'gazetteer.pickle'), 'wb') as f:
        cPickle.dump(gazetteer, f)

    print('Saving sloppy gazetteer dictionary', file=sys.stderr)
    sloppy_gazetteer = defaultdict(set)

    for gazette in gazetteer:
        for word in gazette.split():
            sloppy_gazetteer[word].add(gazette)

    with open(os.path.join(output_dir, 'sloppy_gazetteer.pickle'), 'wb') as f:
        cPickle.dump(sloppy_gazetteer, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=unicode)
    parser.add_argument("output_dir", type=unicode)
    parser.add_argument("experiment_kind", type=unicode)

    args = parser.parse_args()

    collect_gazeteers(args.input_dir, args.output_dir, args.experiment_kind)
