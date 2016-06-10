#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function

import cPickle as pickle
import os
import sys
from bs4 import BeautifulSoup
from collections import namedtuple
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm


urls_dir = sys.argv[1]
resources_dir = sys.argv[2]
TOTAL_DOCS = int(sys.argv[3])
LINK_CHANGE_TOKEN = "LINK_CHANGE_TOKEN"
TITLE_CHANGE_TOKEN = "TITLE_CHANGE_TOKEN"
URLEntity = namedtuple('URLEntity', ['wikipage', 'yagouri', 'categories'])
conll_columns = ("_\t"*8).rstrip()
url_entities = {}
uris_urls = {}
conll_doc_idx = 0


def url_entity_to_string(entity):
    wikipage = entity.wikipage.replace("http://en.wikipedia.org/wiki/", "")
    yagouri = entity.yagouri.replace("http://yago-knowledge.org/resource/", "")
    return "{}#{}#{}".format(wikipage, yagouri, "|".join(sorted(entity.categories)))


print("Loading urls from pickle files", file=sys.stderr)
for pkl in tqdm(sorted(os.listdir(urls_dir))):
    category_name, _ = pkl.strip().split("-0.pickle", 1)
    with open(os.path.join(urls_dir, pkl), "rb") as fp:
        for entity in pickle.load(fp)[1:]:
            yagouri, wikipage = entity
            if wikipage in url_entities:
                url_entities[wikipage].categories.append(category_name)
            else:
                url_entities[wikipage] = URLEntity(
                    wikipage=wikipage,
                    yagouri=yagouri,
                    categories=[category_name]
                )

print("Loading parsed uris", file=sys.stderr)
with open("../../resources/parsed_uris.txt", "r") as f:
    for line in tqdm(f.readlines()):
        line = line.strip().split(",http://", 1)
        uris_urls[line[0]] = "http://{}".format(line[1])

print("Translating documents to column format", file=sys.stderr)
for f in os.listdir(os.path.join(resources_dir, "docs_for_ner")):
    fpath = os.path.join(resources_dir, "docs_for_ner", f)
    if os.path.isfile(fpath):
        os.unlink(fpath)

with open(os.path.join(resources_dir, "docs_for_ner.txt"), "r") as fi:
    for doc_idx, line_doc in tqdm(enumerate(fi), total=TOTAL_DOCS):
        if doc_idx % 60000 == 0:
            conll_doc_idx += 1
        with open(os.path.join(resources_dir, "docs_for_ner/doc_{:02d}.conll".format(conll_doc_idx)), "a") as fo:
            doc_soup = BeautifulSoup(line_doc.decode("utf-8").strip(), "lxml")
            doc_url = doc_soup.doc["url"]
            doc_title = doc_soup.doc.find("h1", {"id": "title"}).replaceWith(" {}. ".format(TITLE_CHANGE_TOKEN))
            doc_links = doc_soup.doc.findAll(lambda tag: tag.name == "a" and "href" in tag.attrs)[:]
            for a in doc_soup.doc.findAll(lambda tag: tag.name == "a" and "href" in tag.attrs):
                a.replaceWith(" {} ".format(LINK_CHANGE_TOKEN))
            document = doc_soup.doc.text
            sentences = sent_tokenize(document)

            for sentence_idx, sentence in enumerate(sentences, start=1):
                token_idx = 0
                for token in word_tokenize(
                        sentence.replace("\xa0", " ").replace("{}.".format(TITLE_CHANGE_TOKEN, TITLE_CHANGE_TOKEN))
                ):
                    if token == LINK_CHANGE_TOKEN:
                        token_tag = doc_links.pop(0)

                        try:
                            entity = url_entities[uris_urls[token_tag["href"]]]
                        except KeyError:
                            entity = None
                        except BaseException as e:
                            print("Document {} had unexpected exception for token {}: {}".format(doc_url, token_idx, e),
                                  file=sys.stderr)
                            entity = None

                        for subtoken_idx, subtoken in enumerate(word_tokenize(token_tag.text)):
                            token_idx += 1
                            if entity is not None and subtoken_idx == 0:
                                print("{} {}\t{}\tB-{}".format(token_idx, subtoken, conll_columns,
                                                               url_entity_to_string(entity)).encode("utf-8"), file=fo)
                            elif entity is not None:
                                print("{} {}\t{}\tI-{}".format(token_idx, subtoken, conll_columns,
                                                               url_entity_to_string(entity)).encode("utf-8"), file=fo)
                            else:
                                print("{} {}\t{}\tO".format(token_idx, subtoken, conll_columns).encode("utf-8"),
                                      file=fo)
                    elif token == TITLE_CHANGE_TOKEN:
                        if doc_url in url_entities:
                            entity = url_entities[doc_url]
                        else:
                            entity = None

                        for subtoken_idx, subtoken in enumerate(word_tokenize(doc_title.text)):
                            token_idx += 1
                            if entity is not None and subtoken_idx == 0:
                                print("{} {}\t{}\tB-{}".format(token_idx, subtoken, conll_columns,
                                                               url_entity_to_string(entity)).encode("utf-8"), file=fo)
                            elif entity is not None:
                                print("{} {}\t{}\tI-{}".format(token_idx, subtoken, conll_columns,
                                                               url_entity_to_string(entity)).encode("utf-8"), file=fo)
                            else:
                                print("{} {}\t{}\tO".format(token_idx, subtoken, conll_columns).encode("utf-8"),
                                      file=fo)
                    else:
                        token_idx += 1
                        print("{} {}\t{}\tO".format(token_idx, token, conll_columns).encode("utf-8"), file=fo)

                print("", file=fo)

            if len(doc_links) != 0:
                with open("logs/errors.log", "a") as ef:
                    print("{} {}".format(doc_idx, doc_url), file=ef)
