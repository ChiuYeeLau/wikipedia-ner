#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals, print_function

import sys
import nltk
from tqdm import tqdm

sentence = []

with open(sys.argv[1], "r") as fi:
    with open(sys.argv[2], "w") as fo:
        for line in tqdm(fi, total=int(sys.argv[3])):
            line = line.decode("utf-8").strip().split()
            if len(line) == 11:
                sentence.append(line)
            elif len(line) == 0:
                tags = nltk.pos_tag([w[1] for w in sentence])

                if len(tags) != len(sentence):
                    sentence = []
                    continue

                for i, w in enumerate(sentence):
                    w[3] = tags[i][1]
                    w[4] = tags[i][1]
                    w[5] = w[-1]

                print("\n".join(["\t".join(w[:6]) for w in sentence]).encode("utf-8"), end="\n\n", file=fo)
                sentence = []
