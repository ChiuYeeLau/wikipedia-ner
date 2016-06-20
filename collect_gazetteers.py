# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import cPickle
import os
import sys
from collections import defaultdict
from tqdm import tqdm
from wikipedianer.corpus.parser import WikipediaCorpusColumnParser

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
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    gazetteer = defaultdict(int)

    for corpus_file in os.listdir(input_dir):
        corpus_doc, _ = corpus_file.split(".", 1)
        print('Getting gazettes for corpus {}'.format(corpus_doc), file=sys.stderr)

        parser = WikipediaCorpusColumnParser(os.path.join(input_dir, corpus_file), remove_stop_words=True)

        for sentence in tqdm(parser, total=FILES_SENTENCES[corpus_file]):
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
