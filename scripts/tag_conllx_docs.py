#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals, print_function

import os
import sys
import nltk
from tqdm import tqdm

sentence = []
header = ""

with open(sys.argv[1], "r") as fi:
    with open(sys.argv[2], "w") as fo:
        for line in tqdm(fi, total=int(sys.argv[3])):
            line = line.decode("utf-8").strip().split()
            if len(line) == 2:
                header = " ".join(line)
            elif len(line) == 11:
                sentence.append(line)
            elif len(line) == 0:
                tags = nltk.pos_tag([w[1] for w in sentence])

                if len(tags) != len(sentence):
                    sentence = []
                    header = ""
                    continue

                for i, w in enumerate(sentence):
                    w[3] = tags[i][1]
                    w[4] = tags[i][1]
                    w[5] = w[-1]

                print(header.encode("utf-8"), file=fo)
                print("\n".join(["\t".join(w[:6]) for w in sentence]).encode("utf-8"), end="\n\n", file=fo)
                sentence = []
                header = ""
