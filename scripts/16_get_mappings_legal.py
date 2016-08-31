#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import cPickle
import os

from collections import defaultdict

URI_YAGO = 'http://yago-knowledge.org/resource/'
entity_mappings = defaultdict(set)
category_mappings = {
    'wordnet_legal_document_106479665': 'no_person',
    'wordnet_due_process_101181475': 'no_person',
    'wordnet_law_108441203': 'no_person',
    'wordnet_law_100611143': 'no_person',
    'wordnet_law_106532330': 'no_person',
    'wordnet_legal_code_106667792': 'no_person',
    'wordnet_criminal_record_106490173': 'no_person',
    'wordnet_legal_power_105198427': 'no_person',
    'wordnet_jurisdiction_108590369': 'no_person',
    'wordnet_judiciary_108166318': 'no_person',
    'wordnet_pleading_106559365': 'no_person',
    'wordnet_court_108329453': 'no_person',
    'wordnet_lawyer_110249950': 'wordnet_person_100007846',
    'wordnet_judge_110225219': 'wordnet_person_100007846',
    'wordnet_adjudicator_109769636': 'wordnet_person_100007846',
    'wordnet_party_110402824': 'wordnet_person_100007846'
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=unicode)
    parser.add_argument('output_file', type=unicode)

    args = parser.parse_args()

    for entity in os.listdir(args.input_dir):
        if not entity.startswith('wordnet'):
            continue
        with open(os.path.join(args.input_dir, entity), 'rb') as f:
            entity_pickle = cPickle.load(f)

        entity_name = entity.split('-0.pickle')[0]

        for yago_uri, wiki_uri in entity_pickle[1:]:
            yago_uri = yago_uri.replace(URI_YAGO, '')

            entity_mappings[yago_uri].add(entity_name)
            entity_mappings[yago_uri].add(category_mappings[entity_name])

    with open(args.output_file, "wb") as f:
        cPickle.dump(entity_mappings, f)
