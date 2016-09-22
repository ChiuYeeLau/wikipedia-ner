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
import os

from wikipedianer.corpus.base import Word
from wikipedianer.corpus.parser import WikipediaCorpusColumnParser

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
        new_line = u'{}\t{}\t{}\n'.format(word.token, word.tag, target)
        output_file.write(new_line.encode("utf-8"))
    output_file.write('\n')


def main():
    """Preprocess the dataset"""
    args = read_arguments()
    parser = WikipediaCorpusColumnParser(file_path=args.input_path)
    filename = os.path.basename(args.input_path)
    with open(os.path.join(args.output_dirname, filename), 'w') as output_file:
        for document in parser:
            if document.has_named_entity:
                write_document(document, output_file, args.target_field)


if __name__ == '__main__':
    main()
