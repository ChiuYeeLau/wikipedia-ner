#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, unicode_literals

import os
import sys
from tqdm import tqdm 


with open(sys.argv[1], "r") as fp, open(sys.argv[2], "r") as fh:
    print(fh.readline(), end="")

    for line in tqdm(fp, total=int(sys.argv[3])):
        print(line, end="")

        if line.decode("utf-8").strip() == "":
            print(fh.readline(), end="")
