# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import re
from collections import OrderedDict


def ner_label_replace(labels, mappings):
    for label in labels:
        if label.startswith('O'):
            yield 'O'
        else:
            yield 'I'


def ne_person_label_replace(labels, mappings):
    for label in labels:
        label = re.sub(r'^[BI]-', '', label)
        if 'no_person' in mappings.get(label, set()):
            yield 'no_person'
        elif 'wordnet_person_100007846' in mappings.get(label, set()):
            yield 'wordnet_person_100007846'
        else:
            yield 'O'


def ne_category_label_replace_legal(labels, mappings):
    for label in labels:
        label = re.sub(r'^[BI]-', '', label)
        if 'wordnet_law_108441203' in mappings.get(label, set()):
            yield 'wordnet_law'
        elif 'wordnet_law_100611143' in mappings.get(label, set()):
            yield 'wordnet_law'
        elif 'wordnet_law_106532330' in mappings.get(label, set()):
            yield 'wordnet_law'
        elif 'wordnet_legal_document_106479665' in mappings.get(label, set()):
            yield 'wordnet_legal_document'
        elif 'wordnet_due_process_101181475' in mappings.get(label, set()):
            yield 'wordnet_due_process'
        elif 'wordnet_legal_code_106667792' in mappings.get(label, set()):
            yield 'wordnet_legal_code'
        elif 'wordnet_criminal_record_106490173' in mappings.get(label, set()):
            yield 'wordnet_criminal_record'
        elif 'wordnet_legal_power_105198427' in mappings.get(label, set()):
            yield 'wordnet_legal_power'
        elif 'wordnet_jurisdiction_108590369' in mappings.get(label, set()):
            yield 'wordnet_jurisdiction'
        elif 'wordnet_judiciary_108166318' in mappings.get(label, set()):
            yield 'wordnet_judiciary'
        elif 'wordnet_pleading_106559365' in mappings.get(label, set()):
            yield 'wordnet_pleading'
        elif 'wordnet_court_108329453' in mappings.get(label, set()):
            yield 'wordnet_court'
        elif 'wordnet_judge_110225219' in mappings.get(label, set()):
            yield 'wordnet_judge'
        elif 'wordnet_adjudicator_109769636' in mappings.get(label, set()):
            yield 'wordnet_adjudicator'
        elif 'wordnet_lawyer_110249950' in mappings.get(label, set()):
            yield 'wordnet_lawyer'
        elif 'wordnet_party_110402824' in mappings.get(label, set()):
            yield 'wordnet_party'
        else:
            yield 'O'


def ne_category_label_replace_movies(labels, mappings):
    for label in labels:
        label = re.sub(r'^[BI]-', '', label)
        if 'wordnet_movie_106613686' in mappings.get(label, set()):
            yield 'wordnet_movie_106613686'
        elif 'wordnet_soundtrack_104262969' in mappings.get(label, set()):
            yield 'wordnet_soundtrack_104262969'
        elif 'wordnet_actor_109765278' in mappings.get(label, set()):
            yield 'wordnet_actor_109765278'
        elif 'wordnet_film_director_110088200' in mappings.get(label, set()):
            yield 'wordnet_film_director_110088200'
        elif 'wordnet_film_maker_110088390' in mappings.get(label, set()):
            yield 'wordnet_film_maker_110088390'
        else:
            yield 'O'


def ne_uri_label_replace(labels, mappings):
    for label in labels:
        yield re.sub(r"^B-", "I-", label)


LABELS_REPLACEMENT = {
    'legal': OrderedDict([
        ("NER", ner_label_replace),
        ("NEP", ne_person_label_replace),
        ("NEC", ne_category_label_replace_legal),
        ("NEU", ne_uri_label_replace)
    ]),
    'movies': OrderedDict([
        ("NER", ner_label_replace),
        ("NEP", ne_person_label_replace),
        ("NEC", ne_category_label_replace_movies),
        ("NEU", ne_uri_label_replace)
    ])
}

LEGAL_SENTENCES = {
    "doc_01.conll": 2962737,
    "doc_02.conll": 3156576,
    "doc_03.conll": 2574401,
    "doc_04.conll": 2379707,
    "doc_05.conll": 2495369,
    "doc_06.conll": 2493490,
    "doc_07.conll": 475036,
    "doc_08.conll": 2994167,
}

MOVIES_SENTENCES = {
    "doc_01.conll": 2636728,
    "doc_02.conll": 2643458,
    "doc_03.conll": 2148683,
    "doc_04.conll": 1821729,
    "doc_05.conll": 1664229,
    "doc_06.conll": 1747290,
    "doc_07.conll": 1872077,
    "doc_08.conll": 1900873,
    "doc_09.conll": 1555085,
    "doc_10.conll": 540151,
    "doc_11.conll": 2678258,
}

SENTENCES = {
    "legal": LEGAL_SENTENCES,
    "movies": MOVIES_SENTENCES,
}

LEGAL_LINES = {
    "doc_01.conll": 81758432,
    "doc_02.conll": 84858928,
    "doc_03.conll": 67968957,
    "doc_04.conll": 62523339,
    "doc_05.conll": 66490686,
    "doc_06.conll": 66268338,
    "doc_07.conll": 12469212,
    "doc_08.conll": 81758180
}

MOVIES_LINES = {
    "doc_01.conll": 73038794,
    "doc_02.conll": 70566294,
    "doc_03.conll": 56058376,
    "doc_04.conll": 47034568,
    "doc_05.conll": 42751351,
    "doc_06.conll": 45453359,
    "doc_07.conll": 49179807,
    "doc_08.conll": 49409121,
    "doc_09.conll": 39881841,
    "doc_10.conll": 13709285,
    "doc_11.conll": 73035225
}

LINES = {
    "legal": LEGAL_LINES,
    "movies": MOVIES_LINES
}
