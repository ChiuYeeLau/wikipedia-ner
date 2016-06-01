#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals, print_function

import sys
from tqdm import tqdm


conll_columns = ("_\t"*8).rstrip()

sentence = []
header = ""
last_header = ""

with open(sys.argv[1], "r") as fi:
    with open(sys.argv[2], "w") as fo:
        for line in tqdm(fi, total=int(sys.argv[3])):
            line = line.decode("utf-8").strip().split()
            if len(line) == 2:
                header = " ".join(line)
            elif len(line) == 14:
                if header == "":
                    header = last_header
                word = "{} {} {} {}".format(line[0], line[1], conll_columns, line[-1])
                sentence.append(word)
            elif len(line) == 0:
                print(header.encode("utf-8"), file=fo)
                print("\n".join(sentence).encode("utf-8"), end="\n\n", file=fo)
                last_header = header
                header = ""
                sentence = []
