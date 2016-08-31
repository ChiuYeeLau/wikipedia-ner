#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function

import argparse
import nltk
import os
import sys
import unicodedata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('documents',
                        type=unicode,
                        nargs='+',
                        help='Documents to tag')
    parser.add_argument('output',
                        type=unicode,
                        help='Directory to save the tagged documents.')

    args = parser.parse_args()

    for document in args.documents:
        print('Tagging document {}'.format(document), file=sys.stderr)
        output = os.path.join(args.output, os.path.basename(document))

        assert output != document, "The input and output document are the same"

        with open(document, 'r') as fi, open(output, 'w') as fo:
            doc_text = fi.read().decode('utf-8').split('\n')

            for paragraph in doc_text:
                if paragraph.strip() == '':
                    continue

                tagged_sentences = nltk.pos_tag_sents([
                    nltk.word_tokenize(unicodedata.normalize("NFKC", sentence))
                    for sentence in nltk.sent_tokenize(paragraph)
                ])

                for sentence in tagged_sentences:
                    for token_idx, (token, pos_tag) in enumerate(sentence, start=1):
                        print('{} {} {} O _ _'.format(token_idx, token, pos_tag).encode('utf-8'), file=fo)

                    print('', file=fo)
