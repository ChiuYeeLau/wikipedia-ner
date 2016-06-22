#!/usr/bin/env bash

source /home/ccardellino/nlp/bin/activate
python parse_corpus_to_matrix_legal.py /home/ccardellino/wikipedia/resources/legal/docs_for_ner /home/ccardellino/wikipedia/resources/legal/features /home/ccardellino/wikipedia/resources/legal/instances/ --stopwords
