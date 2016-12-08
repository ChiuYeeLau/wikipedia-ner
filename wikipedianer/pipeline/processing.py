# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import gensim
import numpy as np
import pickle
import sys

from collections import Counter, defaultdict
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm
from wikipedianer.corpus.parser import InstanceExtractor, WikipediaCorpusColumnParser, WordVectorsExtractor
from wikipedianer.dataset.preprocess import StratifiedSplitter
from wikipedianer.pipeline.util import CL_ITERATIONS, traverse_directory


def collect_gazeteers(path, output_file_path, file_pattern='*.conll'):
    gazetteer = defaultdict(int)

    for file_path in sorted(traverse_directory(path, file_pattern)):
        print('Getting gazetteer for file {}'.format(file_path), file=sys.stderr)

        parser = WikipediaCorpusColumnParser(file_path)

        for sentence in tqdm(parser):
            if sentence.has_named_entity:
                for gazette, value in sentence.get_gazettes().items():
                    gazetteer[gazette] += value

    print('Getting sloppy gazetteer dictionary', file=sys.stderr)
    sloppy_gazetteer = defaultdict(set)

    for gazette in gazetteer:
        for word in gazette.split():
            sloppy_gazetteer[word].add(gazette)

    print('Saving gazetteers', file=sys.stderr)

    with open(output_file_path, 'wb') as f:
        pickle.dump((gazetteer, sloppy_gazetteer), f)

    return gazetteer, sloppy_gazetteer


def feature_selection(dataset, features_names, matrix_file_path, features_file_path, max_features=12000):
    print('Calculating variance of dataset features', file=sys.stderr)
    dataset = csc_matrix(dataset)
    square_dataset = dataset.copy()
    square_dataset.data **= 2
    variance = np.asarray(square_dataset.mean(axis=0) - np.square(dataset.mean(axis=0)))[0]

    print('Getting top %d features' % max_features, file=sys.stderr)
    top_features = np.argsort(variance)[::-1][:max_features]
    min_variance = variance[top_features][-1]

    print('Min variance: %.2e. Getting features over min variance.' % min_variance, file=sys.stderr)
    valid_indices = np.where(variance > min_variance)[0]

    print('Final features count: %d/%d' % (valid_indices.shape[0], dataset.shape[1]))

    print('Filtering features', file=sys.stderr)
    dataset = csr_matrix(dataset[:, valid_indices])

    print('Saving dataset to file {}'.format(matrix_file_path), file=sys.stderr)
    np.savez_compressed(matrix_file_path, data=dataset.data, indices=dataset.indices,
                        indptr=dataset.indptr, shape=dataset.shape)

    print('Saving filtered features names', file=sys.stderr)
    features_names = np.array(features_names)
    filtered_features_names = list(features_names[valid_indices])

    with open(features_file_path, 'wb') as f:
        pickle.dump(filtered_features_names, f)


def parse_corpus_to_handcrafted_features(path, matrix_file_path, labels_file_path, features_file_path,
                                         file_pattern='*.conll', remove_stopwords=False, gazetteer=set(),
                                         sloppy_gazetteer=set(), valid_indices=set()):
    instance_extractor = InstanceExtractor(
        token=True,
        current_tag=True,
        affixes=True,
        max_ngram_length=6,
        prev_token=True,
        next_token=True,
        disjunctive_left_window=4,
        disjunctive_right_window=4,
        tag_sequence_window=2,
        gazetteer=gazetteer,
        sloppy_gazetteer=sloppy_gazetteer,
        valid_indices=valid_indices
    )

    instances = []
    labels = []
    word_index = 0

    for fidx, file_path in enumerate(sorted(traverse_directory(path, file_pattern)), start=1):
        print('Getting instances from corpus {}'.format(file_path), file=sys.stderr)

        parser = WikipediaCorpusColumnParser(file_path, remove_stopwords)

        for sentence in tqdm(parser):
            if sentence.has_named_entity:
                sentence_instances, sentence_labels, word_index = \
                    instance_extractor.get_instances_for_sentence(sentence, word_index)

                instances.extend(sentence_instances)

                for sentence_label in sentence_labels:
                    uri_label, yago_labels, lkif_labels, entity_labels, ner_label = sentence_label
                    ner_tag = uri_label.split('-', 1)[0]

                    if ner_tag != 'O':
                        # Randomize the selection of labels in higher levels
                        label_item = np.random.randint(len(yago_labels))

                        labels.append((
                            '%s' % uri_label,
                            '%s-%s' % (ner_tag, yago_labels[label_item]),
                            '%s-%s' % (ner_tag, lkif_labels[label_item]),
                            '%s-%s' % (ner_tag, entity_labels[label_item]),
                            '%s' % ner_label
                        ))
                    else:
                        labels.append(('O', 'O', 'O', 'O', 'O'))

        if fidx % 3 == 0 and fidx < 27:
            print('Saving partial matrix and labels', file=sys.stderr)

            vectorizer = DictVectorizer(dtype=np.int32)
            dataset_matrix = vectorizer.fit_transform(instances)
            del vectorizer

            np.savez_compressed(matrix_file_path, data=dataset_matrix.data, indices=dataset_matrix.indices,
                                indptr=dataset_matrix.indptr, shape=dataset_matrix.shape)
            del dataset_matrix

            with open(labels_file_path, 'wb') as f:
                pickle.dump(labels, f)

    vectorizer = DictVectorizer(dtype=np.int32)
    dataset_matrix = vectorizer.fit_transform(instances)
    features_names = sorted(vectorizer.vocabulary_, key=vectorizer.vocabulary_.get)
    del vectorizer

    print('Saving features to file %s' % features_file_path, file=sys.stderr)
    with open(features_file_path, 'wb') as f:
        pickle.dump(features_names, f)

    print('Saving final matrix to file %s' % matrix_file_path, file=sys.stderr)
    np.savez_compressed(matrix_file_path, data=dataset_matrix.data, indices=dataset_matrix.indices,
                        indptr=dataset_matrix.indptr, shape=dataset_matrix.shape)

    print('Saving final labels to file %s' % labels_file_path, file=sys.stderr)
    with open(labels_file_path, 'wb') as f:
        pickle.dump(labels, f)

    return dataset_matrix, labels, features_names


def parse_corpus_to_word_vectors(path, matrix_file_path, word_vectors_file, labels_file_path=None,
                                 file_pattern='*.conll', remove_stopwords=False, valid_indices=set(),
                                 window=3, debug=False):
    print('Loading vectors', file=sys.stderr)
    if debug:
        word2vec_model = gensim.models.Word2Vec()
    else:
        word2vec_model = gensim.models.Word2Vec.load_word2vec_format(word_vectors_file, binary=True)

    instance_extractor = WordVectorsExtractor(
        word2vec_model, window, valid_indices
    )

    instances = []
    labels = []
    word_index = 0

    for fidx, file_path in enumerate(sorted(traverse_directory(path, file_pattern)), start=1):
        print('Getting instances from corpus {}'.format(file_path), file=sys.stderr)

        parser = WikipediaCorpusColumnParser(file_path, remove_stopwords)

        for sentence in tqdm(parser):
            if sentence.has_named_entity:
                sentence_instances, sentence_labels, word_index = \
                    instance_extractor.get_instances_for_sentence(sentence, word_index)

                instances.extend(sentence_instances)

                if labels_file_path is not None:
                    uri_label, yago_labels, lkif_labels, entity_labels, ner_label = sentence_labels
                    ner_tag = uri_label.split('-', 1)[0]

                    if ner_tag != 'O':
                        # Randomize the selection of labels in higher levels
                        label_item = np.random.randint(len(yago_labels))

                        labels.append((
                            '%s' % uri_label,
                            '%s-%s' % (ner_tag, yago_labels[label_item]),
                            '%s-%s' % (ner_tag, lkif_labels[label_item]),
                            '%s-%s' % (ner_tag, entity_labels[label_item]),
                            '%s' % ner_label
                        ))
                    else:
                        labels.append(('O', 'O', 'O', 'O', 'O'))

        if fidx % 3 == 0 and fidx < 27:
            print('Saving partial matrix and labels', file=sys.stderr)

            np.savez_compressed(matrix_file_path, dataset=np.vstack(instances))
            if labels_file_path is not None:
                with open(labels_file_path, 'wb') as f:
                    pickle.dump(labels, f)

    print('Saving final matrix to file %s' % matrix_file_path, file=sys.stderr)
    np.savez_compressed(matrix_file_path, dataset=np.vstack(instances))

    if labels_file_path is not None:
        print('Saving final labels to file %s' % labels_file_path, file=sys.stderr)
        with open(labels_file_path, 'wb') as f:
            pickle.dump(labels, f)

    return labels


def split_dataset(labels, indices_save_path, classes_save_path, train_size=0.8,
                  test_size=0.1, validation_size=0.1, min_count=10):
    classes = {}

    print('Getting uri labels', file=sys.stderr)
    uri_labels = [label[0] for label in labels]

    print('Getting filtered classes', file=sys.stderr)
    filtered_classes = {l for l, v in Counter(uri_labels).items() if v >= min_count}

    print('Getting filtered indices', file=sys.stderr)
    filtered_indices = np.array([i for i, l in enumerate(uri_labels)
                                 if (l != 'O' and l in filtered_classes) or (l == 'O')], dtype=np.int32)

    strat_split = StratifiedSplitter(np.array(uri_labels), filtered_indices)

    print('Splitting the dataset', file=sys.stderr)
    train_indices, test_indices, validation_indices = strat_split.get_splitted_dataset_indices(
        train_size=train_size, test_size=test_size, validation_size=validation_size)

    print('Saving indices to file %s' % indices_save_path, file=sys.stderr)
    np.savez_compressed(indices_save_path, train_indices=train_indices, test_indices=test_indices,
                        validation_indices=validation_indices, filtered_indices=filtered_indices)

    for idx, iteration in enumerate(CL_ITERATIONS):
        print('Getting classes for iteration %s' % iteration, file=sys.stderr)
        replaced_labels = [label[idx] for label in labels]
        classes[iteration] = np.unique(np.array(replaced_labels)[filtered_indices], return_counts=True)

    print('Saving classes to file %s' % classes_save_path, file=sys.stderr)
    with open(classes_save_path, 'wb') as f:
        pickle.dump(classes, f)


def subsample_non_entities(path, output_file_path, file_pattern='*.conll', remove_stopwords=False):
    labels = []

    for file_path in sorted(traverse_directory(path, file_pattern)):
        print('Parsing %s' % file_path, file=sys.stderr)

        parser = WikipediaCorpusColumnParser(file_path, remove_stopwords)

        for sentence in tqdm(parser):
            if sentence.has_named_entity:
                labels.extend(sentence.labels)

    print('All corpora parsed', file=sys.stderr)
    print('Counting labels', file=sys.stderr)

    unique_labels, inverse_indices, counts = np.unique(labels, return_inverse=True, return_counts=True)
    counts.sort()
    subsample_count = min(counts[:-1].sum(), counts[-1])
    nne_index = np.where(unique_labels == 'O')[0][0]
    nne_instances = np.random.permutation(np.where(inverse_indices == nne_index)[0])[:subsample_count]
    ne_instances = np.where(inverse_indices != nne_index)[0]

    valid_indices = set(nne_instances).union(set(ne_instances))

    print('Saving indices to {}'.format(output_file_path), file=sys.stderr)
    with open(output_file_path, 'wb') as f:
        pickle.dump(valid_indices, f)

    return valid_indices


