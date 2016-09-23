"""
Script to process corpus to be used by the StanfordNER system.

The output follows the CoNLL format, with columns separated by tabs. The
column content is word, postag and target.

The target is specified in the arguments and it can be the name of any
attibute of the wikipedianer.corpus.base.Word class.

For example, to train the Stanford NER system to predict the YAGO class of
entities, the target will be 'wordnet_categories'. On the other hand,
only to recognize named entities, the target will be 'ner_tag' or 'is_ner'.

If the target attribute is multiple, them the first target will be selected
"""

import argparse
import numpy
import pickle
import os

from contextlib import nested
from operator import itemgetter

from wikipedianer.corpus.base import Word
from wikipedianer.corpus.parser import WikipediaCorpusColumnParser
from wikipedianer.dataset.preprocess import labels_filterer
from wikipedianer.dataset.preprocess import StratifiedSplitter

DEFAULT_TARGET = 'O'

TAG_PROCESS_FUNCTIONS = {
    # Merge tags I and B
    'ner_tag': lambda tag: 'I' if not tag.startswith('O') else DEFAULT_TARGET
}

def read_arguments():
    """Parses the arguments from the stdin and returns an object."""
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dirname', type=unicode,
                        help='Path of directory with the files to preprocess')
    parser.add_argument('target_field', type=unicode,
                        help='Name of the attibute of the'
                             'wikipedianer.corpus.base.Word to use as third'
                             'column of the output')
    parser.add_argument('--output_dirname', '-o', type=unicode,
                        help='Name of the directory to save the output file')
    parser.add_argument('--splits', type=float, nargs=3,
                        help='Proportions of entities to include in training, '
                             'testing and evaluation partitions. For example '
                             '0.70 0.20 0.10')

    return parser.parse_args()


class StanfordPreprocesser(object):
    """Class to preprocess the dataset to train the Stanford NER system.

    The process cycle consists in:
        -- Read all files in input_dirname
        -- Filter documents without a named entity
        -- Filter classes with less than 3 examples in total.
        -- Split the dataset according to the proportions in splits with an
            stratified strategy.
        -- Save the splits into files inside output dirname.
    """
    
    def __init__(self, input_dirname, target_field, output_dirname, splits):
        self.input_dirname = input_dirname
        self.target_field = target_field
        self.output_dirname = output_dirname
        self.splits = splits if splits else []

        # Lists indices of filtered documents and their corresponding labels.
        # If a document has multiple labels, one is selected randomly.
        self.labels = []
        self.documents = []

        # List with the filenames used to construct the current state
        # All files in input_dirname
        filenames = sorted(filter(
            lambda f: os.path.isfile(os.path.join(self.input_dirname, f)),
            os.listdir(self.input_dirname)))
        self.file_paths = [os.path.join(self.input_dirname, filename)
                           for filename in filenames]

        # Indices of the documents for each split in the corpus
        self.train_doc_index = None
        self.test_doc_index = None
        self.validation_doc_index = None

    def preprocess(self):
        """Runs all the preprocess tasks."""
        self.read_documents()
        self.filter_labels()
        self.split_corpus()
        self.write_splits()

    def add_label(self, document):
        """Adds the target field from the document to the labels list."""
        labels_in_document = document.get_unique_properties(self.target_field)
        assert len(labels_in_document) >= 1
        # TODO(mili) do something better
        self.labels.append(labels_in_document.pop())

    def read_documents(self):
        """Adds all documents and labels to the inputs labels and documents."""
        current_document_index = 0
        for file_path in self.file_paths:
            print "Reading file: {}".format(file_path)
            parser = WikipediaCorpusColumnParser(file_path=file_path)
            for document in parser:
                if document.has_named_entity:
                    self.documents.append(current_document_index)
                    self.add_label(document)
                current_document_index += 1

    def get_target(self, word):
        """Returns a processed target for the word."""
        target = getattr(word, self.target_field)
        if isinstance(target, list):
            target = target[0] if len(target) > 0 else DEFAULT_TARGET
        if target is None or target == u'':
            target = DEFAULT_TARGET
        if self.target_field in TAG_PROCESS_FUNCTIONS:
            target = TAG_PROCESS_FUNCTIONS[self.target_field](target)
        return target

    def filter_labels(self):
        """Filter the labels and documents with less than 3 occurrences"""
        filtered_indices = labels_filterer(self.labels)
        self.labels = numpy.array(self.labels)[filtered_indices]
        self.documents = [
            doc_index for index, doc_index in enumerate(self.documents)
            if index in filtered_indices]

    def split_corpus(self):
        """Splits dataset into train, test and validation."""
        print "Splitting dataset."
        splitter = StratifiedSplitter(self.labels)
        # This split returns the filtered indexes of self.labels (equivalent to
        # self.documents) corresponding to each split. These are not absolute
        # document indices
        train_index, test_index, validation_index = (
            splitter.get_splitted_dataset_indices(*self.splits))

        if not len(train_index) or not len(test_index):
            raise ValueError("ERROR not enough instances to split")

        self.train_doc_index = [
            doc_index for index, doc_index in enumerate(self.documents)
            if index in train_index]
        self.test_doc_index = [
            doc_index for index, doc_index in enumerate(self.documents)
            if index in test_index]
        self.validation_doc_index = [
            doc_index for index, doc_index in enumerate(self.documents)
            if index in validation_index]

    def write_document(self, document, output_file):
        """Writes the document into the output file with proper format."""
        for word in document:
            if not hasattr(word, self.target_field):
                print 'Warning: skipping word {} without target field'.format(
                    word)
                continue
            target = self.get_target(word)
            new_line = u'{}\t{}\t{}\n'.format(word.token, word.tag, target)
            output_file.write(new_line.encode("utf-8"))
        output_file.write('\n')

    def write_splits(self):
        """Re-reads the input files and writes documents into the split files.
        """
        if not self.output_dirname:
            return
        print "Writing {} documents".format(len(self.documents))
        print "Train dataset size {}".format(len(self.train_doc_index))
        print "Test dataset size {}".format(len(self.test_doc_index))
        print "Validation dataset size {}".format(len(self.validation_doc_index))
        current_document_index = 0

        train_filename = os.path.join(self.output_dirname, 'train.conll')
        test_filename = os.path.join(self.output_dirname, 'test.conll')
        val_filename = os.path.join(self.output_dirname, 'validation.conll')
        with nested(open(train_filename, 'w'), open(test_filename, 'w'),
                    open(val_filename, 'w')) as (train_f, test_f, val_f):
            for file_path in self.file_paths:
                parser = WikipediaCorpusColumnParser(file_path=file_path)
                for document in parser:
                    if current_document_index in self.train_doc_index:
                        self.write_document(document, train_f)
                    elif current_document_index in self.test_doc_index:
                        self.write_document(document, test_f)
                    elif current_document_index in self.validation_doc_index:
                        self.write_document(document, val_f)
                    current_document_index += 1

        # Save indices to file
        indices_filename = os.path.join(self.output_dirname,
                                        'split_indices.pickle')
        with open(indices_filename, 'w') as indices_file:
            pickle.dump((self.train_doc_index, self.test_doc_index,
                         self.validation_doc_index), indices_file)


def main():
    """Preprocess the dataset"""
    # TODO(mili) Filter O occurrences?
    args = read_arguments()
    processer = StanfordPreprocesser(args.input_dirname, args.target_field,
                                     args.output_dirname, args.splits)

    processer.preprocess()


if __name__ == '__main__':
    main()
