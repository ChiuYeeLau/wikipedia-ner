"""Black box tests for the preprocess_for_stanford.py script.

It's a little bit hacky but it's fast to try out. The test file contains:
    4 documents with label wordnetA
    3 documents with label wordnetB
    2 documents without label
    1 document with label wordnetC

wordnetA is the only person category.
"""

import csv
import os
import subprocess
import unittest
import utils

from collections import defaultdict
from preprocess_for_stanford import DocumentsFilter
from wikipedianer.corpus.parser import WikipediaCorpusColumnParser

TEST_DIRNAME = 'test_files/stanford_preprocess/'
OUTPUT_DIRNAME = os.path.join(TEST_DIRNAME, 'generated')

CATEGORY_A = 'wordnet_adjudicator_109769636'
CATEGORY_B = 'wordnet_criminal_record_106490173'
CATEGORY_C = 'wordnet_court_108329453'

PERSONS = [CATEGORY_A]


class DocumentMock(object):
    def __init__(self):
        self.tags = []


class StanfordPreprocessTests(unittest.TestCase):
    """Tests for the preprocess_for_stanford.py script.

    Runs the script taking an input file and reads the output files.
    """
    @staticmethod
    def _read_documents(filename):
        documents = []
        current_document = DocumentMock()
        file_path = os.path.join(OUTPUT_DIRNAME, filename)
        with open(file_path, 'r') as input_file:
            reader = csv.reader(input_file, delimiter='\t')
            for line in reader:
                if len(line) == 0:
                    documents.append(current_document)
                    current_document = DocumentMock()
                else:
                    current_document.tags.append(line[-1])
        return documents

    @staticmethod
    def run_script(task_name):
        """Runs the script"""
        utils.safe_mkdir(OUTPUT_DIRNAME)
        command = ('python preprocess_for_stanford.py'
                   ' {} {} -o {} --splits 0.7 0.3 0.0'.format(
                        TEST_DIRNAME, task_name, OUTPUT_DIRNAME))
        subprocess.check_call(command, shell=True)

    def tearDown(self):
        """Removes any created file"""
        utils.safe_rmdir(OUTPUT_DIRNAME)

    def test_filter_sentences_no_ner(self):
        """Sentences without a named entity are filtered out."""
        self.run_script('yago')
        train_documents = self._read_documents('train.conll')
        self.assertEqual(5, len(train_documents))
        for document in train_documents:
            self.assertLessEqual(1, len(set(document.tags)))

        test_documents = self._read_documents('test.conll')
        self.assertEqual(2, len(test_documents))
        for document in test_documents:
            self.assertLessEqual(1, len(set(document.tags)))

    def test_category_names(self):
        """Test if the category names are correct."""
        self.run_script('yago')
        counts = defaultdict(lambda: 0)
        train_documents = self._read_documents('train.conll')
        for document in train_documents:
            for category in set(document.tags):
                counts[category] += 1

        self.assertIn(CATEGORY_A, counts)
        self.assertIn(CATEGORY_B, counts)
        self.assertNotIn(CATEGORY_C, counts)

    def test_ner(self):
        """Test if the classes for NER are correct."""
        self.run_script('ner')

        train_documents = self._read_documents('train.conll')
        test_documents = self._read_documents('test.conll')
        self.assertEqual(8, len(train_documents) + len(test_documents))

        self.assertEqual(6, len(train_documents))
        self.assertEqual(2, len(test_documents))
        for document in train_documents:
            self.assertEqual(3, len(set(document.tags)))

        for document in test_documents:
            self.assertEqual(3, len(set(document.tags)))


class DocumentsFilterTests(unittest.TestCase):
    """Test suite for DocumentFilter class."""

    FILTERED_OUTPUT_PATH = os.path.join(TEST_DIRNAME,
                                        DocumentsFilter.OUTPUT_DIRNAME)

    def tearDown(self):
        """Removes any created file"""
        utils.safe_rmdir(self.FILTERED_OUTPUT_PATH)

    def test_read_filtered_files(self):
        input_filepath = os.path.join(TEST_DIRNAME,
                                      'test_stanford_preprocess.conll')
        # Read original documents
        parser = WikipediaCorpusColumnParser(file_path=input_filepath)
        original_documents = []
        for document in parser:
            if document.has_named_entity:
                original_documents.append(document)

        # Filter documents
        filter_ = DocumentsFilter(TEST_DIRNAME)
        filter_.filter_documents() 

        # Re-read new documents
        filtered_filepath = os.path.join(self.FILTERED_OUTPUT_PATH,
                                         'test_stanford_preprocess.conll')
        # Read original documents
        parser = WikipediaCorpusColumnParser(file_path=filtered_filepath)
        for index, document in enumerate(parser):
            self.assertTrue(document.has_named_entity)
            self.assertEqual(original_documents[index], document)


if __name__ == '__main__':
    unittest.main()
