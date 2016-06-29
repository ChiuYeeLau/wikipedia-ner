# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import LINES


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=unicode)
    parser.add_argument("output_file", type=unicode)
    parser.add_argument("documents", type=unicode)
    parser.add_argument("--experiment_kind", type=unicode, default='legal')
    parser.add_argument("--sample", type=int, default=100)

    args = parser.parse_args()

    print('Loading documents from file {}'.format(args.documents), file=sys.stderr)
    with open(args.documents, "r") as f:
        documents = [doc.decode('utf-8').strip() for doc in f.readlines()]

    print('Selecting {} documents at random'.format(args.sample), file=sys.stderr)
    chosen_documents_indices = np.random.permutation(len(documents))[:args.sample]
    chosen_documents = set()
    for doc_idx in chosen_documents_indices:
        chosen_documents.add(documents[doc_idx])

    for conll_file in sorted(os.listdir(args.input_dir)):
        corpus_doc, _ = conll_file.split(".", 1)
        print('Getting selected documents from corpus {}'.format(corpus_doc), file=sys.stderr)

        last_doc = ''
        last_doc_in_sample = False

        with open(os.path.join(args.input_dir, conll_file), 'r') as fi, open(args.output_file, 'a') as fo:
            for line in tqdm(fi, total=LINES[args.experiment_kind][conll_file]):
                line = line.decode('utf-8').strip()

                if line != '':
                    _, token, tag, class_string, head, dep = line.split()

                    if class_string.endswith('-DOC'):
                        last_doc_in_sample = class_string in chosen_documents

                if last_doc_in_sample:
                    print(line, file=fo)
