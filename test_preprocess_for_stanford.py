"""Black box tests for the preprocess_for_stanford.py script.

It's a little bit hacky but it's fast to try out. The test file contains:
    4 documents with label wordnetA
    3 documents with label wordnetB
    2 documents without label
    1 document with label wordnetC
"""

import csv
import os
import subprocess
import unittest
import utils

from collections import defaultdict

TEST_DIRNAME = 'test_files'
OUTPUT_DIRNAME = os.path.join(TEST_DIRNAME, 'generated')

CATEGORY_A = 'wordnet_adjudicator_109769636'
CATEGORY_B = 'wordnet_criminal_record_106490173'
CATEGORY_C = 'wordnet_court_108329453'


class DocumentMock(object):
    def __init__(self):
        self.tags = []


class StanfordPreprocessTests(unittest.TestCase):
    """Tests for the preprocess_for_stanford.py script.

    Runs the script taking an input file and reads the output files.
    """
    @staticmethod
    def _read_documents(file_path):
        documents = []
        current_document = DocumentMock()
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
    def run_script(target):
        """Runs the script"""
        utils.safe_mkdir(OUTPUT_DIRNAME)
        command = ('python preprocess_for_stanford.py'
                   ' {} {} -o {} --splits 0.7 0.3 0.0'.format(
                        TEST_DIRNAME, target, OUTPUT_DIRNAME))
        subprocess.check_call(command, shell=True)

    def tearDown(self):
        """Removes any created file"""
        utils.safe_rmdir(OUTPUT_DIRNAME)

    def test_filter_sentences_no_ner(self):
        """Sentences without a named entity are filtered out."""
        self.run_script('wordnet_categories')
        train_documents = self._read_documents(
            os.path.join(OUTPUT_DIRNAME, 'train.conll'))
        self.assertEqual(5, len(train_documents))
        for document in train_documents:
            self.assertLessEqual(1, len(set(document.tags)))

        test_documents = self._read_documents(
            os.path.join(OUTPUT_DIRNAME, 'test.conll'))
        self.assertEqual(2, len(test_documents))
        for document in test_documents:
            self.assertLessEqual(1, len(set(document.tags)))

    def test_category_names(self):
        """Test if the category names are correct."""
        self.run_script('wordnet_categories')
        counts = defaultdict(lambda: 0)
        train_documents = self._read_documents(
            os.path.join(OUTPUT_DIRNAME, 'train.conll'))
        for document in train_documents:
            for category in set(document.tags):
                counts[category] += 1

        self.assertIn(CATEGORY_A, counts)
        self.assertIn(CATEGORY_B, counts)
        self.assertNotIn(CATEGORY_C, counts)

    def test_ner(self):
        """Test if the classes for NER are correct."""
        self.run_script('ner_tag')

        train_documents = self._read_documents(
            os.path.join(OUTPUT_DIRNAME, 'train.conll'))
        self.assertEqual(6, len(train_documents))
        for document in train_documents:
            # Only tag I and O
            self.assertEqual(2, len(set(document.tags)))

        test_documents = self._read_documents(
            os.path.join(OUTPUT_DIRNAME, 'test.conll'))
        self.assertEqual(2, len(test_documents))
        for document in test_documents:
            self.assertEqual(2, len(set(document.tags)))


if __name__ == '__main__':
    unittest.main()
