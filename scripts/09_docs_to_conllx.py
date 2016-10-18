#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to convert plain text wikipedia documents to conll format.

The documents in the file resources_dir/docs_for_ner.txt are processed using
as labels the entites in urls_dir.
"""

from __future__ import unicode_literals, print_function

import argparse
import cPickle as pickle
import os
import sys
from bs4 import BeautifulSoup
from collections import namedtuple
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm


LINK_CHANGE_TOKEN = "LINK_CHANGE_TOKEN"
TITLE_CHANGE_TOKEN = "TITLE_CHANGE_TOKEN"
URLEntity = namedtuple('URLEntity', ['wikipage', 'yagouri', 'categories'])
conll_columns = ("_\t"*8).rstrip()


def parse_arguments():
    """Returns the stdin arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--urls_dirpath', type=unicode)
    parser.add_argument('-r', '--resources_dir', type=unicode)
    parser.add_argument('-t', '--total_docs', type=int)

    return parser.parse_args()


def url_entity_to_string(entity):
    wikipage = entity.wikipage.replace("http://en.wikipedia.org/wiki/", "")
    yagouri = entity.yagouri.replace("http://yago-knowledge.org/resource/", "")
    return "{}#{}#{}".format(wikipage, yagouri,
                             "|".join(sorted(entity.categories)))


def load_urls(urls_dirpath):
    """Returns a dictionary from the wikipage to a URLEntity."""
    url_entities = {}
    for pkl in tqdm(sorted(os.listdir(urls_dirpath))):
        category_name, _ = pkl.strip().split("-0.pickle", 1)
        with open(os.path.join(urls_dirpath, pkl), "rb") as fp:
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
    return url_entities


def load_parsed_uris(resources_dir):
    """Reads the clean uris and returns a map from uris to urls."""
    uris_urls = {}
    input_filepath = os.path.join(resources_dir, 'parsed_uris.txt')
    with open(input_filepath, "r") as input_file:
        for line in tqdm(input_file.readlines()):
            line = line.strip().split(",http://", 1)
            uris_urls[line[0]] = "http://{}".format(line[1])
    return uris_urls


def create_directory(resources_dir):
    """Creates a directory for the result files"""
    for f in os.listdir(os.path.join(resources_dir, "docs_for_ner")):
        fpath = os.path.join(resources_dir, "docs_for_ner", f)
        if os.path.isfile(fpath):
            os.unlink(fpath)

def write_link_token(token_idx, entity, token_tag, output_file):
    """Writes to output file the LINK_CHANGE_TOKEN in conll format."""
    for subtoken_idx, subtoken in enumerate(word_tokenize(token_tag.text)):
        token_idx += 1
        if entity is not None and subtoken_idx == 0:
            print("{} {}\t{}\tB-{}".format(
                token_idx, subtoken, conll_columns,
                url_entity_to_string(entity)).encode("utf-8"), file=output_file)
        elif entity is not None:
            print("{} {}\t{}\tI-{}".format(
                token_idx, subtoken, conll_columns,
                url_entity_to_string(entity)).encode("utf-8"), file=output_file)
        else:
            print("{} {}\t{}\tO".format(
                token_idx, subtoken, conll_columns).encode("utf-8"),
                file=output_file)
    return token_idx


def write_title_token(token_idx, doc_title, entity, output_file):
    """Writes to output file the TITLE_CHANGE_TOKEN in conll format."""
    for subtoken_idx, subtoken in enumerate(word_tokenize(doc_title.text)):
        token_idx += 1
        if entity is not None and subtoken_idx == 0:
            print("{} {}\t{}\tB-{}-DOC".format(
                token_idx, subtoken, conll_columns,
                url_entity_to_string(entity)).encode("utf-8"), file=output_file)
        elif entity is not None:
            print("{} {}\t{}\tI-{}-DOC".format(
                token_idx, subtoken, conll_columns,
                url_entity_to_string(entity)).encode("utf-8"), file=output_file)
        else:
            print("{} {}\t{}\tO-DOC".format(
                token_idx, subtoken, conll_columns).encode("utf-8"),
                file=output_file)
    return token_idx


def transform_document(output_file, url_entities, uris_urls, line_doc):
    """Preprocess and adds document to output_file."""
    doc_soup = BeautifulSoup(line_doc.decode("utf-8").strip(), "lxml")
    doc_url = doc_soup.doc["url"]
    doc_title = doc_soup.doc.find("h1", {"id": "title"}).replaceWith(
        " {}. ".format(TITLE_CHANGE_TOKEN))
    doc_links = doc_soup.doc.findAll(
        lambda tag: tag.name == "a" and "href" in tag.attrs)[:]
    for a in doc_soup.doc.findAll(
        lambda tag: tag.name == "a" and "href" in tag.attrs):
        a.replaceWith(" {} ".format(LINK_CHANGE_TOKEN))
    document = doc_soup.doc.text
    sentences = sent_tokenize(document)

    for sentence in sentences:
        token_idx = 0
        sentence = sentence.replace("\xa0", " ").replace(
            "{}.".format(TITLE_CHANGE_TOKEN), TITLE_CHANGE_TOKEN)
        for token in word_tokenize(sentence):
            if token == LINK_CHANGE_TOKEN:
                token_tag = doc_links.pop(0)

                try:
                    entity = url_entities[uris_urls[token_tag["href"]]]
                except KeyError:
                    entity = None
                except BaseException as e:
                    print("Document {} had unexpected exception for token {}: {}".format(
                        doc_url, token_idx, e), file=sys.stderr)
                entity = None

                token_idx = write_link_token(token_idx, entity, token_tag,
                                             output_file)
            elif token == TITLE_CHANGE_TOKEN:
                entity = None
                if doc_url in url_entities:
                    entity = url_entities[doc_url]
                token_idx = write_title_token(token_idx, doc_title, entity,
                                              output_file)
            else:
                token_idx += 1
                print("{} {}\t{}\tO".format(
                    token_idx, token, conll_columns).encode("utf-8"),
                    file=output_file)

        print("", file=output_file)

    if len(doc_links) != 0:
        with open("logs/errors.log", "a") as error_file:
            print("{} {}".format(doc_idx, doc_url), file=error_file)


def transform_to_conll(resources_dir, total_docs, url_entities, uris_urls):
    """Transform docs_for_ner.txt in resources_dir to conll format."""
    input_filepath = os.path.join(resources_dir, "docs_for_ner.txt")
    conll_doc_idx = 0
    with open(input_filepath, "r") as input_file:
        for doc_idx, line_doc in tqdm(enumerate(input_file), total=total_docs):
            if doc_idx % 60000 == 0:
                conll_doc_idx += 1
            output_filepath = os.path.join(
                resources_dir,
                "docs_for_ner/doc_{:02d}.conll".format(conll_doc_idx))
            with open(output_filepath, "a") as output_file:
                transform_document(output_file, url_entities, uris_urls,
                                   line_doc)


def main():
    """Main script function."""
    args = parse_arguments()

    print("Loading urls from pickle files", file=sys.stderr)
    url_entities = load_urls(args.urls_dirpath)
    print("Loading parsed uris", file=sys.stderr)
    uris_urls = load_parsed_uris(args.resources_dir)
    create_directory(args.resources_dir)
    print("Translating documents to column format", file=sys.stderr)
    transform_to_conll(args.resources_dir, args.total_docs, url_entities,
                       uris_urls)



if __name__ == '__main__':
    main()
