# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import re


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
            yield 'wordnet_law_108441203'
        elif 'wordnet_law_100611143' in mappings.get(label, set()):
            yield 'wordnet_law_100611143'
        elif 'wordnet_law_106532330' in mappings.get(label, set()):
            yield 'wordnet_law_106532330'
        elif 'wordnet_legal_document_106479665' in mappings.get(label, set()):
            yield 'wordnet_legal_document_106479665'
        elif 'wordnet_due_process_101181475' in mappings.get(label, set()):
            yield 'wordnet_due_process_101181475'
        elif 'wordnet_legal_code_106667792' in mappings.get(label, set()):
            yield 'wordnet_legal_code_106667792'
        elif 'wordnet_criminal_record_106490173' in mappings.get(label, set()):
            yield 'wordnet_criminal_record_106490173'
        elif 'wordnet_legal_power_105198427' in mappings.get(label, set()):
            yield 'wordnet_legal_power_105198427'
        elif 'wordnet_jurisdiction_108590369' in mappings.get(label, set()):
            yield 'wordnet_jurisdiction_108590369'
        elif 'wordnet_judiciary_108166318' in mappings.get(label, set()):
            yield 'wordnet_judiciary_108166318'
        elif 'wordnet_pleading_106559365' in mappings.get(label, set()):
            yield 'wordnet_pleading_106559365'
        elif 'wordnet_court_108329453' in mappings.get(label, set()):
            yield 'wordnet_court_108329453'
        elif 'wordnet_judge_110225219' in mappings.get(label, set()):
            yield 'wordnet_judge_110225219'
        elif 'wordnet_adjudicator_109769636' in mappings.get(label, set()):
            yield 'wordnet_adjudicator_109769636'
        elif 'wordnet_lawyer_110249950' in mappings.get(label, set()):
            yield 'wordnet_lawyer_110249950'
        elif 'wordnet_party_110402824' in mappings.get(label, set()):
            yield 'wordnet_party_110402824'
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
    'legal': dict(
        NER=ner_label_replace,
        NEP=ne_person_label_replace,
        NEC=ne_category_label_replace_legal,
        NEU=ne_uri_label_replace
    ),
    'movies': dict(
        NER=ner_label_replace,
        NEP=ne_person_label_replace,
        NEC=ne_category_label_replace_movies,
        NEU=ne_uri_label_replace
    )
}
