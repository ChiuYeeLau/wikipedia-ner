# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import cPickle
import os
import sys
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

    for corpus_file in os.listdir(input_dir):
        corpus_doc, _ = corpus_file.split(".", 1)
        print('Getting gazettes for corpus {}'.format(corpus_doc), file=sys.stderr)

        clean_gazettes = {}
        sloppy_gazettes = {}

        parser = WikipediaCorpusColumnParser(os.path.join(input_dir, corpus_file), remove_stop_words=True)

        for sentence in tqdm(parser, total=FILES_SENTENCES[corpus_file]):
            if sentence.has_named_entity:
                sentence_clean_gazettes = sentence.get_clean_gazettes()
                sentence_sloppy_gazettes = sentence.get_sloppy_gazettes()

                for clean_gazette in sentence_clean_gazettes:
                    if clean_gazette in clean_gazettes:
                        clean_gazettes[clean_gazette].update(sentence_clean_gazettes[clean_gazette])
                    else:
                        clean_gazettes[clean_gazette] = sentence_clean_gazettes[clean_gazette]

                for sloppy_gazette in sentence_sloppy_gazettes:
                    if sloppy_gazette in sloppy_gazettes:
                        sloppy_gazettes[sloppy_gazette].update(sentence_sloppy_gazettes[sloppy_gazette])
                    else:
                        sloppy_gazettes[sloppy_gazette] = sentence_sloppy_gazettes[sloppy_gazette]

        print('Saving gazetteer pickles', file=sys.stderr)

        with open(os.path.join(output_dir, 'clean_gazettes_{}.pickle'.format(corpus_doc)), 'wb') as f:
            cPickle.dump(clean_gazettes, f)

        with open(os.path.join(output_dir, 'sloppy_gazettes_{}.pickle'.format(corpus_doc)), 'wb') as f:
            cPickle.dump(sloppy_gazettes, f)
