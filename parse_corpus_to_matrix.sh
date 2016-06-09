#!/usr/bin/env bash

source /home/ccardellino/nlp/bin/activate
python parse_corpus_to_matrix_raw.py /home/ccardellino/wikipedia/resources/docs_for_ner /home/ccardellino/wikipedia/resources/features /home/ccardellino/wikipedia/resources/instances/ --stopwords
