# -*- coding: utf-8 -*-
"""Tests for the query_babelfy script functions.

To obtain the sample response run use the following senteces.
[u'The Bantu Investment Corporation Act, Act No 34 of 1959, formed part of the apartheid system of racial segregation in South Africa. In combination with the Bantu Homelands Development Act of 1965, it allowed the South African government to capitalize on entrepreneurs operating in the Bantustans. It created a Development Corporation in each of the Bantustans.',
u'At the end of the trial, the prosecutor asked for the acquittal of all of the accused persons. The defence renounced its right to plead, preferring to observe a minute of silence in favor of François Mourmand, who had died in prison during remand. Yves Bot, general prosecutor of Paris, came to the trial on its last day, without previously notifying the president of the Cour d\'assises, Mrs. Mondineu-Hederer; while there, Bot presented his apologies to the defendants on behalf of the legal system—he did this before the verdict was delivered, taking for granted a "not guilty" ruling, for which some magistrates reproached him afterwards.',
u'The affair caused public indignation and questions about the general workings of justice in France. The role of an inexperienced magistrate, Fabrice Burgaud,[5] fresh out of the Ecole Nationale de la Magistrature was underscored, as well as the undue weight given to children\'s words and to psychiatric expertise, both of which were revealed to have been wrong.'
    ]
"""

import unittest
import json

import query_babelfy


class EvaluateFromBabelfyResponseTests(unittest.TestCase):
    """Mocks the babefy response and checks the resulting metrics."""

    TEST_FILEPATH = 'test_files/sample_response.json'
    
    def setUp(self):
        with open(self.TEST_FILEPATH, 'r') as test_file:
            self.response = json.load(test_file)

    def test_process_prediction(self):
        """Check the array of predictions generated from Babelfy response."""
        predictions, classes = query_babelfy.process_prediction(
            self.response, 248)
        self.assertEqual(248, len(predictions))
        expected = ['O'] * 248
        prefix = 'http://dbpedia.org/resource/'
        for index in range(1, 5):
            expected[index] = prefix + 'Bantu_Investment_Corporation_Act,_1959'
        for index in range(39, 42):
            expected[index] = prefix + 'South_Africa'
        expected[49] = prefix + 'Bantustan'
        for index in range(111, 113):
            expected[index] = prefix + 'Yves_Bot'
        expected[139] = prefix + 'Hillary_Rodham_Clinton'
        for index in range(198, 201):
            expected[index] = prefix + 'Judiciary_of_France'
        for index in range(219, 224):
            expected[index] = (prefix + 
                               'French_National_School_for_the_Judiciary')
        self.assertEqual(expected, predictions)
        self.assertEqual(set(expected), classes)


if __name__ == '__main__':
    unittest.main()