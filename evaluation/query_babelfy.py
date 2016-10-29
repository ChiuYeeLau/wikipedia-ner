# -*- coding: utf-8 -*-

"""
Script to query the http API of Babelfy.

Note: babelfy has a limit of 1000 queries per day.
"""

import argparse
import csv
import gzip
import json
import logging
logging.basicConfig(level=logging.INFO)
import tqdm
import urllib2
import urllib

from prediction_document import PredictionDocument
from SPARQLWrapper import SPARQLWrapper, JSON
from StringIO import StringIO


BABELFY_ENDPOINT_URL = 'https://babelfy.io/v1/disambiguate'
YAGO_ENPOINT_URL = 'https://linkeddata1.calcul.u-psud.fr/sparql'
DOCUMENT_SEPARATOR = u'-' * 100


# TODO(mili) use this from utils.py
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


def read_arguments():
    """Parses the arguments from the stdin and returns an object."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filepath', type=unicode,
                        help='Path of the file to preprocess')
    parser.add_argument('--output_filepath', type=unicode,
                        help='Path to save the predictions')
    parser.add_argument('--key', type=unicode,
                        help='Key for babelfy.')
    parser.add_argument('--task', type=unicode, default='NEU',
                        help='Name of the task to evaluate. Can be NEC or NEU')
    parser.add_argument('--docs', type=int, default='10',
                        help='Number of docs to process.')

    return parser.parse_args()


def send_query_to_babelfy(text, key):
    """Sends a query to the API and returns the response."""
    params = {
        'text' : text.encode('utf-8'),
        'lang' : 'EN',
        'key'  : key,
        'extAIDA': True,
        'annType': 'NAMED_ENTITIES'
    }

    url = BABELFY_ENDPOINT_URL + '?' + urllib.urlencode(params)
    request = urllib2.Request(url)
    request.add_header('Accept-encoding', 'gzip')
    response = urllib2.urlopen(request)

    if response.info().get('Content-Encoding') == 'gzip':
        buf = StringIO(response.read())
        response_file = gzip.GzipFile(fileobj=buf)
        return json.loads(response_file.read())


def query_sparql(query, endpoint):
    """Run a query again an SPARQL endpoint.

    Returns:
        A double list with only the values of each requested variable in
        the query. The first row in the result contains the name of the
        variables.
    """
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)

    sparql.setQuery(query)
    response = sparql.query().convert()
    bindings = response['results']['bindings']
    variables = response['head']['vars']
    result = [variables]
    for binding in bindings:
        row = []
        for variable in variables:
            row.append(binding[variable]['value'])
        result.append(row)
    return result


def query_yago(dbpedia_url):
    """Sends a query to yago to get the uri and the class for dbpedia_url."""
    yago_query = """SELECT DISTINCT ?entity ?className WHERE {
            ?entity rdf:type ?className .
            ?entity <http://www.w3.org/2002/07/owl#sameAs> <%s>
        } LIMIT 100""" % (dbpedia_url,)
    return query_sparql(yago_query, YAGO_ENPOINT_URL)


def get_yago_uri(labels):
    """Gets the equivalent yago uri and classes to the dbpedia urls in labels.
    """
    entity_names = []
    yago_classes = []
    mapped_labels = {'O': {'entity': 'O', 'yago_class': 'O'}}
    for dbpedia_url in labels:
        if dbpedia_url not in mapped_labels:
            # Get yago uri and class from wikipage
            info = query_yago(dbpedia_url)
            if len(info) <= 1:
                logging.info(
                    'No yago entity for dbpedia page {}'.format(dbpedia_url))
                mapped_labels[dbpedia_url] = {
                    'entity': None, 'yago_class': None}
            else:
                yago_class = None
                for _, class_name in info[1:]:
                    class_name = class_name.split('/')[-1]
                    if class_name in NE_CATEGORY_LABEL_LEGAL_MAP:
                        # We select the first result.
                        yago_class = NE_CATEGORY_LABEL_LEGAL_MAP[class_name]
                        break
                mapped_labels[dbpedia_url] = {
                    'entity': u'I-{}'.format(info[1][0].split('/')[-1]),
                    'yago_class': yago_class
                }
        entity_names.append(mapped_labels[dbpedia_url]['entity'])
        yago_classes.append(mapped_labels[dbpedia_url]['yago_class'])
    return entity_names, yago_classes


def process_prediction(data, char_map):
    """Transforms the result from babelfy into an array of classes.

    Each element in the array corresponds to the tag given by babelfy. char_map
    is a list with the start and end index of each word.
    """
    predictions = ['O'] * len(char_map)
    scores = [0] * len(char_map)
    classes = set(['O'])
    char2word = [-1] * (char_map[-1][1] + 1)
    for word_index, (start, end) in enumerate(char_map):
        for index in range(start, end + 1):
            char2word[index] = word_index

    for result in data:
        char_fragment = result.get('charFragment')
        start = char_fragment.get('start')
        end = char_fragment.get('end')
        for word_index in set(char2word[start:end+1]):
            if word_index < 0:
                continue
            if (predictions[word_index] == 'O' and
                    scores[word_index] < result.get('score')):
                predictions[word_index] = result.get('DBpediaURL')
                scores[word_index] = result.get('score')
                classes.add(result.get('DBpediaURL'))

    return predictions, classes


def read_input_file(filename):
    """Reads the input file and generates PredictionDocument instances"""
    with open(filename, 'r') as input_file:
        content = input_file.read()
    documents = content.split(DOCUMENT_SEPARATOR.encode('utf-8'))
    for document in documents:
        if document != '\n' and document != '':
            new_doc = PredictionDocument()
            new_doc.loads(document)
            yield new_doc


def evaluate_prediction(predictions, document):
    """Calculates the metrics for the babelfy prediction.

    Args:
        prediction: list of assigned labels.
        document: an instance of PredictionDocument.
    """
    # We are going to match word by word.

    true_labels = document.tags
    yago_entities, yago_classes = process_prediction(predictions)



def main():
    """Main function of script"""
    args = read_arguments()

    with open(args.output_filepath, 'w') as output_file:
        for doc_index, document in tqdm.tqdm(enumerate(
                read_input_file(args.input_filepath)), total=args.docs):
            result = send_query_to_babelfy(document.text, args.key)
            predictions = process_prediction(result, document.char_map)
            neu_labels, nec_labels = get_yago_uri(predictions[0])
            if len(set([len(neu_labels), len(document.tags),
                        len(nec_labels), len(predictions[0])])) != 1:
                import ipdb; ipdb.set_trace()
                logging.error('Predictions have different sizes!')
                return
            if args.task == 'NEU':
                document.dump(output_file, new_tags=neu_labels)
            else:
                document.dump(output_file, new_tags=nec_labels)
            output_file.write(DOCUMENT_SEPARATOR + u'\n')
            output_file.write(DOCUMENT_SEPARATOR + u'\n')
            if doc_index == args.docs - 1:
                break


if __name__ == '__main__':
    main()
