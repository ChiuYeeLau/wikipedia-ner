#!/usr/bin/env python
# coding: utf-8

import cPickle
import os
import sys
from tqdm import tqdm

sloppy_gazetteers = {}

for pickle in tqdm(sorted(os.listdir("../resources/docs_for_ner_gazetteers"))):
    if pickle.startswith("sloppy"):
        with open(os.path.join("../resources/docs_for_ner_gazetteers", pickle), "rb") as f:
            sg = cPickle.load(f)

        for key in sg:
            if key in sloppy_gazetteers:
                sloppy_gazetteers[key].update(sg[key])
            else:
                sloppy_gazetteers[key] = sg[key]

print "Saving file"
with open("../resources/docs_for_ner_gazetteers/sloppy_gazetteers.pickle", "wb") as f:
    cPickle.dump(sloppy_gazetteers, f)
