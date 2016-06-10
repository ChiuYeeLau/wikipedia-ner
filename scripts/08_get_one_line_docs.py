#!/usr/bin/env python
# coding: utf-8

from __future__ import unicode_literals, print_function

import os
import sys
from bs4 import BeautifulSoup
from tqdm import tqdm


docs_in_ner = {}
last_doc_in_ner = False
is_doc_title = False
working_dir = sys.argv[1]


with open(os.path.join(working_dir, "docs_in_ner.txt"), "r") as f:
    for line in f:
        doc, url = line.decode("utf-8").strip().split(",", 1)
        docs_in_ner[doc] = url

for f in sorted(os.listdir("../../clean")):
    print("Extracting relevant documents for file {}".format(f), file=sys.stderr)
    with open(os.path.join(working_dir, "docs_for_ner.txt"), "a") as fo:
        with open("../../clean/{}".format(f), "r") as fi:
            for line in tqdm(fi):
                line = line.decode("utf-8").strip()
                if line.startswith("<doc"):
                    soup = BeautifulSoup(line, "lxml")
                    try:
                        doc_id = soup.doc["id"]
                    except TypeError:
                        last_doc_in_ner = False
                        continue

                    last_doc_in_ner = doc_id in docs_in_ner

                    if last_doc_in_ner:
                        print("<doc id=\"{}\" url=\"{}\"> ".format(doc_id, docs_in_ner[doc_id]).encode("utf-8"),
                              end="", file=fo)
                        is_doc_title = True
                elif last_doc_in_ner:
                    if line.startswith("</doc>"):
                        print("</doc>".encode("utf-8"), file=fo)
                        last_doc_in_ner = False
                    elif is_doc_title:
                        print("<h1 id=\"title\">{}</h1> ".format(line).encode("utf-8"), end="", file=fo)
                        is_doc_title = False
                    else:
                        print("{} ".format(line).encode("utf-8"), end="", file=fo)
