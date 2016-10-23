# -*- coding: utf-8 -*-
"""Tests for the query_babelfy script functions.

"""

import unittest
import json

import prediction_document
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


class PredictionDocumentTests(unittest.TestCase):
    """Tests for PredictionDocument class"""
    TEST_FILEPATH = 'test_files/sample_document.txt'

    EXPECTED_TEXT = (
        u'The Nicholas Academic Centers provide after‐school tutoring and'
        u' mentoring for high school students in the Santa Ana Unified School '
        u'District. The two Centers—NAC I in downtown Santa Ana ( 412 W. 4th '
        u'Street ), and NAC II on the campus of Valley High'
    )

    def setUp(self):
        with open(self.TEST_FILEPATH, 'r') as test_file:
            self.raw_text = test_file.read()

        self.simple_document = prediction_document.PredictionDocument()
        self.simple_document.loads(self.raw_text)

    def test_load_and_dump(self):
        """Test a dumped file is equal to the original loaded file."""
        output = StringIO.StringIO()
        self.simple_document.dump(output)
        output.seek(0)
        new_content = output.read().split('\n')
        raw_text = self.raw_text.split('\n')
        # The last element is added by the buffer.
        self.assertEqual(raw_text, new_content[:-1])

    def test_generated_text(self):
        """Test if the generated text string is correct."""
        self.assertEqual(self.EXPECTED_TEXT, self.simple_document.text)


if __name__ == '__main__':
    unittest.main()
