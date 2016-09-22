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
import pickle
import os

from operator import itemgetter

from wikipedianer.corpus.base import Word
from wikipedianer.corpus.parser import WikipediaCorpusColumnParser
from wikipedianer.dataset.preprocess import StratifiedSplitter

DEFAULT_TARGET = 'O'


def read_arguments():
    """Parses the arguments from the stdin and returns an object."""
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=unicode,
                        help='Path of file to preprocess')
    parser.add_argument('output_dirname', type=unicode,
                        help='Name of the directory to save the output file')
    parser.add_argument('target_field', type=unicode,
                        help='Name of the attibute of the'
                             'wikipedianer.corpus.base.Word to use as third'
                             'column of the output')
    parser.add_argument('--splits', type=float, nargs=3,
                        help='Proportions of entities to include in training, '
                             'testing and evaluation partitions. For example '
                             '0.70 0.20 0.10')

    return parser.parse_args()


def write_document(document, output_file, target_field):
    """Writes the document into the output file with proper format."""
    for word in document:
        if not hasattr(word, target_field):
            print 'Warning: skipping word {} without target field'.format(word)
            continue
        target = getattr(word, target_field)
        if isinstance(target, list):
            target = target[0] if len(target) > 0 else DEFAULT_TARGET
        if target is None or target == u'':
            target = DEFAULT_TARGET
        # TODO(mili) add exception to merge B- and I- classes
        new_line = u'{}\t{}\t{}\n'.format(word.token, word.tag, target)
        output_file.write(new_line.encode("utf-8"))
    output_file.write('\n')


def write_documents(documents, indexes, filename, target_field):
    """Writes all documents listed in indexes into a file with name filename."""
    with open(filename, 'w') as output_file:
        for document in itemgetter(*indexes)(documents):
            write_document(document, output_file, target_field)


def add_label(document, labels, target_field):
    """Adds the target field from the document to the labels list."""
    labels_in_document = document.get_unique_properties(target_field)
    assert len(labels_in_document) >= 1
    labels.append(labels_in_document.pop())  # TODO(mili) do something better


def read_documents_from_file(input_path, labels, documents, target_field):
    """Adds all documents and labels to the inputs labels and documents."""
    parser = WikipediaCorpusColumnParser(file_path=input_path)
    for document in parser:
        if document.has_named_entity:
            documents.append(document)
            add_label(document, labels, target_field)


def split_corpus(labels, documents, target_field,
                 splits, output_dirname, filename):
    """Splits dataset into train, test and validation. Saves into files."""
    splitter = StratifiedSplitter(labels)
    train_index, test_index, val_index = splitter.get_splitted_dataset_indices(
        *splits, ignore_warnings=True)

    if not len(train_index) or not len(test_index) or not len(val_index):
        print "ERROR not enough instances to split"
        return

    write_documents(documents, train_index,
                    os.path.join(output_dirname, filename) + '-train',
                    target_field)
    write_documents(documents, test_index,
                    os.path.join(output_dirname, filename) + '-test',
                    target_field)
    write_documents(documents, val_index,
                    os.path.join(output_dirname, filename) + '-validation',
                    target_field)

    # Save indices to file
    indices_filename = os.path.join(output_dirname,
                                    filename + '-split_indices.pickle')
    with open(indices_filename, 'w') as indices_file:
        pickle.dump((train_index, test_index, val_index), indices_file)


def main():
    """Preprocess the dataset"""
    # TODO(mili) Read files from directory
    # TODO(mili) Filter O occurrences?
    args = read_arguments()
    filename = os.path.basename(args.input_path)

    # Lists with the sorted filtered documents and their corresponding labels.
    # If a document has multiple labels, one is selected randomly.
    labels = []
    documents_to_write = []

    read_documents_from_file(args.input_path, labels, documents_to_write,
                             args.target_field)

    if not args.splits:
        args.splits = []
    split_corpus(labels, documents_to_write, args.target_field, args.splits,
                 args.output_dirname, filename)


if __name__ == '__main__':
    main()
