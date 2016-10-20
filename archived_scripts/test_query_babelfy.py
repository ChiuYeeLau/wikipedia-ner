# -*- coding: utf-8 -*-
"""Tests for the query_babelfy script functions.

"""

import unittest
import json

import query_babelfy
import StringIO



def get_char_map(original_words, original_text):
    """Build word to character postion map."""
    last_position = 0
    char_map = []
    for word in original_words:
        start_index = original_text[last_position:].find(word) + last_position
        end_index = start_index + len(word) - 1
        char_map.append((start_index, end_index))
        last_position = end_index
    return char_map



class EvaluateFromBabelfyResponseTests(unittest.TestCase):
    """Mocks the babefy response and checks the resulting metrics."""

    TEST_FILEPATH = 'test_files/sample_response.json'
    ORIGINAL_TEXT = ('The Bantu Investment Corporation Act, Act No 34 of '
                     '1959, formed part of the apartheid system of racial '
                     'segregation in South Africa')
    ORIGINAL_WORDS = ('The Bantu Investment Corporation Act , Act No 34 of '
                      '1959 , formed part of the apartheid system of racial '
                      'segregation in South Africa').split()

    def setUp(self):
        self.char_map = get_char_map(self.ORIGINAL_WORDS, self.ORIGINAL_TEXT)
        with open(self.TEST_FILEPATH, 'r') as test_file:
            self.response = json.load(test_file)

    def test_process_prediction(self):
        """Check the array of predictions generated from Babelfy response."""
        predictions, classes = query_babelfy.process_prediction(
            self.response, self.char_map)
        self.assertEqual(len(self.ORIGINAL_WORDS), len(predictions))
        expected = ['O'] * len(self.ORIGINAL_WORDS)
        prefix = 'http://dbpedia.org/resource/'
        for index in range(1, 5):
            expected[index] = prefix + 'Bantu_Investment_Corporation_Act,_1959'
        expected[23] = prefix + 'Africa'
        self.assertEqual(expected, predictions)
        self.assertEqual(set(expected), classes)


class SimpleDocumentTests(unittest.TestCase):
    """Tests for SimpleDocument class"""
    TEST_FILEPATH = 'test_files/sample_document.txt'

    def test_load_and_dump(self):
        """Test a dumped file is equal to the original loaded file."""
        with open(self.TEST_FILEPATH, 'r') as test_file:
            raw_text = test_file.read()

        simple_document = query_babelfy.SimpleDocument()
        simple_document.loads(raw_text)
        output = StringIO.StringIO()
        simple_document.dumps(output)
        self.assertEqual(raw_text, output)


if __name__ == '__main__':
    unittest.main()
