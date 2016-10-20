# -*- coding: utf-8 -*-

"""
Script to query the http API of Babelfy.

Note: babelfy has a limit of 1000 queries per day.
"""

import argparse
import csv
import gzip
import json
import urllib2
import urllib

from SPARQLWrapper import SPARQLWrapper, JSON
from StringIO import StringIO


BABELFY_ENDPOINT_URL = 'https://babelfy.io/v1/disambiguate'
YAGO_ENPOINT_URL = 'https://linkeddata1.calcul.u-psud.fr/sparql'
DOCUMENT_SEPARATOR = '-' * 100


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
    parser.add_argument('--task', type=unicode,
                        help='Name of the task to evaluate. Can be NEC or NEU')

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
                print 'No yago entity for dbpedia page {}', dbpedia_url
                mapped_labels[dbpedia_url] = {
                    'entity': None, 'yago_class': None}
                continue
            yago_class = None
            for _, class_name in info[1:]:
                class_name = class_name.split('/')[-1]
                if class_name in NE_CATEGORY_LABEL_LEGAL_MAP:
                    # We select the first result.
                    yago_class = NE_CATEGORY_LABEL_LEGAL_MAP[class_name]
                    break
            mapped_labels[dbpedia_url] = {
                'entity': info[1][0].split('/')[-1],
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


class SimpleDocument(object):
    def __init__(self):
        # List of sentences. Each sentence is a list of pairs (word, postag)
        self.sentences = []
        # List with original labels. One label per word in sentences.
        self.tags = []
        # Representation of document as continous text.
        self.text = ''
        # First rows of document (without processing) that are tagged as title.
        self.title = []
        # Map from indices in self.raw text to indices in self.tags
        self.char_map = []
        self.start_sentence()

    def loads(self, text):
        """Construct document from raw text in column format."""
        lines = text.split('\n')
        # Read title
        for line in lines:
            if len(line.split('\t')) == 5 and line.endswith('DOCUMENT START'):
                self.title.append(line)
            else:
                break
        # Read words
        for line in lines[len(self.title) + 1:]:
            if line == '\n':
                self.start_sentence()
            else:
                word, pos_tag, tag = line.split('\t')[1:4]
                self.add_word(word, pos_tag, tag)
                if word not in ['.', '?', '!', ',', '"', '\'']:
                    self.text += ' '
                self.char_map.append(
                    (len(self.text), len(self.text) + len(word) - 1))
                self.text += word

    def dumps(self, output_file, new_tags=None):
        """Write document in original format to output_file.

        Appends document to output_file."""
        # Write title
        for title_line in self.title:
            output_file.write(title_line + '\n')
        output_file.write('\n')
        # Write content
        if not new_tags:
            new_tags = self.tags
        tag_index = 0
        for sentence in self.sentences:
            for word_index, word in enumerate(sentence):
                output_file.write('{}\t{}\t{}\t{}\n'.format(
                    word_index, word[0], word[1], new_tags[tag_index]))
                tag_index += 1
            output_file.write('\n')

    def start_sentence(self):
        """Records the start of a new sentence in document"""
        self.sentences.append([])

    def add_word(self, word, pos_tag, tag):
        """Add word to document."""
        self.sentences[-1].append((word, pos_tag))
        self.tags.append(tag)


def read_input_file(filename):
    """Reads the input file and generates SimpleDocument instances"""
    with open(filename, 'r') as input_file:
        content = input_file.read()
    documents = content.split(DOCUMENT_SEPARATOR)
    for document in documents:
        if document != '\n':
            new_doc = SimpleDocument()
            new_doc.read_from_text(document)
            yield new_doc


def evaluate_prediction(predictions, document):
    """Calculates the metrics for the babelfy prediction.

    Args:
        prediction: list of assigned labels.
        document: an instance of SimpleDocument.
    """
    # We are going to match word by word.

    true_labels = document.tags
    yago_entities, yago_classes = process_prediction(predictions)



def main():
    """Main function of script"""
    args = read_arguments()

    # text = [u'The Bantu Investment Corporation Act, Act No 34 of 1959, formed part of the apartheid system of racial segregation in South Africa. In combination with the Bantu Homelands Development Act of 1965, it allowed the South African government to capitalize on entrepreneurs operating in the Bantustans. It created a Development Corporation in each of the Bantustans.',
    #     u'At the end of the trial, the prosecutor asked for the acquittal of all of the accused persons. The defence renounced its right to plead, preferring to observe a minute of silence in favor of François Mourmand, who had died in prison during remand. Yves Bot, general prosecutor of Paris, came to the trial on its last day, without previously notifying the president of the Cour d\'assises, Mrs. Mondineu-Hederer; while there, Bot presented his apologies to the defendants on behalf of the legal system—he did this before the verdict was delivered, taking for granted a "not guilty" ruling, for which some magistrates reproached him afterwards.',
    #     u'The affair caused public indignation and questions about the general workings of justice in France. The role of an inexperienced magistrate, Fabrice Burgaud,[5] fresh out of the Ecole Nationale de la Magistrature was underscored, as well as the undue weight given to children\'s words and to psychiatric expertise, both of which were revealed to have been wrong.'
    #     ]
    # result = send_query_to_babelfy('\n'.join(text), args.key)


    import ipdb; ipdb.set_trace()

    print get_yago_uri([
        'http://dbpedia.org/resource/Bantu_Investment_Corporation_Act,_1959',
        'http://dbpedia.org/resource/BabelNet',
        'http://dbpedia.org/resource/Gun_laws_in_Virginia',
        'http://dbpedia.org/resource/Mauricio_Macri'])



if __name__ == '__main__':
    main()
