# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import os
import re
import shutil
import subprocess

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


def ne_category_label_replace_legal_lkif(labels, mappings):
    for label in labels:
        label = re.sub(r'^[BI]-', '', label)
        if 'Code' in mappings.get(label, set()):
            yield 'Code'
        elif 'Hohfeldian_Power' in mappings.get(label, set()):
            yield 'Hohfeldian_Power'
        elif 'Legal_Document' in mappings.get(label, set()):
            yield 'Legal_Document'
        elif 'Legal_Speech_Act' in mappings.get(label, set()):
            yield 'Legal_Speech_Act'
        elif 'Legislative_Body' in mappings.get(label, set()):
            yield 'Legislative_Body'
        elif 'Professional_Legal_Role' in mappings.get(label, set()):
            yield 'Professional_Legal_Role'
        elif 'Regulation' in mappings.get(label, set()):
            yield 'Regulation'
        else:
            yield 'O'


NE_CATEGORY_LABEL_LEGAL_MAP = {
    'wordnet_law_108441203': 'wordnet_law',
    'wordnet_law_100611143': 'wordnet_law',
    'wordnet_law_106532330': 'wordnet_law',
    'wordnet_legal_document_106479665': 'wordnet_legal_document',
    'wordnet_due_process_101181475': 'wordnet_due_process',
    'wordnet_legal_code_106667792': 'wordnet_legal_code',
    'wordnet_criminal_record_106490173': 'wordnet_criminal_record',
    'wordnet_legal_power_105198427': 'wordnet_legal_power',
    'wordnet_jurisdiction_108590369': 'wordnet_jurisdiction',
    'wordnet_judiciary_108166318': 'wordnet_judiciary',
    'wordnet_pleading_106559365': 'wordnet_pleading',
    'wordnet_court_108329453': 'wordnet_court',
    'wordnet_judge_110225219': 'wordnet_judge',
    'wordnet_adjudicator_109769636': 'wordnet_adjudicator',
    'wordnet_lawyer_110249950': 'wordnet_lawyer',
    'wordnet_party_110402824': 'wordnet_party',
}


NE_CATEGORY_PERSON_LEGAL_MAP = {
    'wordnet_law_108441203': 'not_person',
    'wordnet_law_100611143': 'not_person',
    'wordnet_law_106532330': 'not_person',
    'wordnet_legal_document_106479665': 'not_person',
    'wordnet_due_process_101181475': 'not_person',
    'wordnet_legal_code_106667792': 'not_person',
    'wordnet_criminal_record_106490173': 'not_person',
    'wordnet_legal_power_105198427': 'not_person',
    'wordnet_jurisdiction_108590369': 'not_person',
    'wordnet_judiciary_108166318': 'not_person',
    'wordnet_pleading_106559365': 'not_person',
    'wordnet_court_108329453': 'not_person',
    'wordnet_judge_110225219': 'person',
    'wordnet_adjudicator_109769636': 'person',
    'wordnet_lawyer_110249950': 'person',
    'wordnet_party_110402824': 'not_person',
}


def ne_category_label_replace_legal_wordnet(labels, mappings):
    for label in labels:
        label = re.sub(r'^[BI]-', '', label)
        mapping = mappings.get(label, set())
        result = None
        for key, value in NE_CATEGORY_LABEL_LEGAL_MAP.iteritems():
            if key in mapping:
                result = value
        yield result if result else 'O'


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
        ("NEC", ne_category_label_replace_legal_wordnet),
        ("LKIF", ne_category_label_replace_legal_lkif),
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


def safe_mkdir(dir_path):
    """Checks if a directory exists, and if it doesn't, creates one."""
    try:
        os.stat(dir_path)
    except OSError:
        os.mkdir(dir_path)


def safe_rmdir(dir_path):
    """Checks if a directory exists, and if it does, removes it."""
    try:
        os.stat(dir_path)
        shutil.rmtree(dir_path)
    except OSError:
        pass


def count_lines_in_file(filepath):
    """Return the number of lines in a file obtained with the command wc -l"""
    args = ['wc', '-l', filepath]
    output = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
    return int(output.split()[0])
