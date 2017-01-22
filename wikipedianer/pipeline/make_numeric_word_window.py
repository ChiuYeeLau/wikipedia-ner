"""Transforms a word window file into an array of index from a model's vocab."""
import argparse
import pickle

import gensim
import logging

import numpy
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def read_arguments():
    """Parses the arguments from the stdin and returns an object."""
    parser = argparse.ArgumentParser()
    parser.add_argument('word_windows_path', type=str,
                        help='Path of pickled file with the word windows.')
    parser.add_argument('model_path', type=str,
                        help='Path of gensim Word2vec model file.')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save the output word windows.')
    parser.add_argument('--binary_save', action='store_true',
                        help='Load the model with binary flag on.')
    return parser.parse_args()


def make_numeric_word_window(windows, model):
    new_windows = numpy.zeros((len(windows), len(windows[0])),
                              dtype=numpy.int32)
    for index1, instance in tqdm(enumerate(windows), total=len(windows)):
        for index2, word in enumerate(instance):
            word_index = 0
            if word[0] in model:
                word_index = model.wv.vocab[word[0]].index
            elif word[1] in model:
                word_index = model.wv.vocab[word[1]].index
            new_windows[index1, index2] = word_index
    return new_windows


def main():
    args = read_arguments()
    if not args.output_file:
        logging.error('Must specify an output file')
        return
    logging.info('Loading model')
    model = gensim.models.Word2Vec.load_word2vec_format(
        args.model_path, binary=args.binary_save)
    logging.info('Loading word windows')
    with open(args.word_windows_path, 'rb') as wordfile:
        windows = pickle.load(wordfile)

    logging.info('Creating instances')
    new_windows = make_numeric_word_window(windows, model)

    logging.info('Saving instances')
    with open(args.output_file, 'wb') as wordfile:
        pickle.dump(new_windows, wordfile)

    logging.info('All operations completed')


if __name__ == '__main__':
    main()
