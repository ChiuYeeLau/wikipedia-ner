"""Test for the function in parse_evaluation_corpus script"""

import unittest
from wikipedianer.corpus.parser import InstanceExtractor, WordVectorsExtractor
from wikipedianer.corpus.parser import WikipediaCorpusColumnParser

from parse_evaluation_corpus import process_sentences
from scipy.sparse import csr_matrix


class TestParseEvaluationCorpus(unittest.TestCase):
    """Test for functions in parse_evaluation_corpus script."""

    TEST_INPUT_FILE = 'test_files/wikipedia_doc.conll'

    def _read_test_file(self):
        feature_set = set()
        original_instances = []
        with open(self.TEST_INPUT_FILE, 'r') as input_file:
            for line in input_file.readlines():
                if len(line) <= 1:
                    continue
                word, tag = line.split()[1:3]
                original_instances.append((word, tag))
                feature_set.add('token:current={}'.format(word))
                feature_set.add('tag:current={}'.format(tag))
        return ({feature: index for index, feature in enumerate(feature_set)},
                original_instances)

    def test_process_sentences(self):
        """Tes the process_sentences matrix output."""

        instance_extractor = InstanceExtractor(
            token=True,
            current_tag=True,
        )
        parser = WikipediaCorpusColumnParser(self.TEST_INPUT_FILE)
        features, original_instances = self._read_test_file()
        matrix = process_sentences(parser,
                                   instance_extractor, features)[0]
        self.assertEqual((len(original_instances), len(features)), matrix.shape)
        self.assertIsInstance(matrix, csr_matrix)
        matrix = matrix.toarray()
        for index, (word, tag) in enumerate(original_instances):
            self.assertGreater(
                matrix[index, features['token:current={}'.format(word)]], 0,
                msg='0 value for token {} at line {}'.format(word, index))
            self.assertGreater(
                matrix[index, features['tag:current={}'.format(tag)]], 0,
                msg='0 value for pos {} at line {}'.format(tag, index))


if __name__ == '__main__':
    unittest.main()
