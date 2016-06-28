# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle
import os
import sys
from collections import defaultdict
from tqdm import tqdm
from wikipedianer.corpus.parser import WikipediaCorpusColumnParser

LEGAL_SENTENCES = {
    "doc_01.conll": 2962737,
    "doc_02.conll": 3156576,
    "doc_03.conll": 2574401,
    "doc_04.conll": 2379707,
    "doc_05.conll": 2495369,
    "doc_06.conll": 2493490,
    "doc_07.conll": 475036,
    "doc_08.conll": 2994167,
}

MOVIES_SENTENCES = {
    "doc_01.conll": 2636728,
    "doc_02.conll": 2643458,
    "doc_03.conll": 2148683,
    "doc_04.conll": 1821729,
    "doc_05.conll": 1664229,
    "doc_06.conll": 1747290,
    "doc_07.conll": 1872077,
    "doc_08.conll": 1900873,
    "doc_09.conll": 1555085,
    "doc_10.conll": 540151,
    "doc_11.conll": 2678258,
}

SENTENCES = {
    "legal": LEGAL_SENTENCES,
    "movies": MOVIES_SENTENCES,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=unicode)
    parser.add_argument("output_dir", type=unicode)
    parser.add_argument("experiment_kind", type=unicode)

    args = parser.parse_args()

    if args.experiment_kind not in SENTENCES:
        print('Not a valid experiment (must be "legal" or "movies")', file=sys.stderr)
        sys.exit(1)

    gazetteer = defaultdict(int)

    for corpus_file in sorted(os.listdir(args.input_dir)):
        corpus_doc, _ = corpus_file.split(".", 1)
        print('Getting gazettes for corpus {}'.format(corpus_doc), file=sys.stderr)

        parser = WikipediaCorpusColumnParser(os.path.join(args.input_dir, corpus_file), remove_stop_words=True)

        for sentence in tqdm(parser, total=SENTENCES[args.experiment_kind][corpus_file]):
            if sentence.has_named_entity:
                for gazette, value in sentence.get_gazettes().iteritems():
                    gazetteer[gazette] += value

    print('Saving gazetteer', file=sys.stderr)
    with open(os.path.join(args.output_dir, 'gazetteer.pickle'), 'wb') as f:
        cPickle.dump(gazetteer, f)

    print('Saving sloppy gazetteer dictionary', file=sys.stderr)
    sloppy_gazetteer = defaultdict(set)

    for gazette in gazetteer:
        for word in gazette.split():
            sloppy_gazetteer[word].add(gazette)

    with open(os.path.join(args.output_dir, 'sloppy_gazetteer.pickle'), 'wb') as f:
        cPickle.dump(sloppy_gazetteer, f)
