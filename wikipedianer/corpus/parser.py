# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import cPickle
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
from .base import Sentence, Word, WORDNET_CATEGORIES_LEGAL, WORDNET_CATEGORIES_MOVIES, YAGO_RELATIONS_MOVIES


STOPWORDS_SET = set(stopwords.words())


class InstanceExtractor(object):
    def __init__(self, **kwargs):
        self.token = bool(kwargs.get('token', False))
        self.current_tag = bool(kwargs.get('current_tag', False))
        self.affixes = bool(kwargs.get('affixes', False))
        self.max_ngram_length = int(kwargs.get('max_ngram_length', 6))
        self.prev_token = bool(kwargs.get('prev_token', False))
        self.next_token = bool(kwargs.get('next_token', False))
        self.disjunctive_left_window = int(kwargs.get('disjunctive_left_window', 0))
        self.disjunctive_right_window = int(kwargs.get('disjunctive_right_window', 0))
        self.tag_sequence_window = int(kwargs.get('tag_sequence_window', 0))
        self.gazetteer = kwargs.get('gazetteer', set())
        self.sloppy_gazetteer = kwargs.get('sloppy_gazetteer', {})
        self.valid_indices = kwargs.get('valid_indices', set())

    def _features_for_word(self, word, sentence, named_entity=None):
        """
        :param word: wikipedianer.corpus.base.Word
        :param sentence: wikipedianer.corpus.base.Sentence
        :param named_entity: wikipedianer.corpus.base.NamedEntity
        """
        features = defaultdict(int)

        if self.token:
            features['token:current={}'.format(word.token)] += 1

        if self.current_tag:
            features['tag:current={}'.format(word.tag)] += 1

        if self.affixes:
            prefixes, suffixes = word.get_affixes(self.max_ngram_length)

            for i in range(len(prefixes)):
                prefix = prefixes[i]
                features['prefix:{}-gram={}'.format(len(prefix), prefix)] += 1

            for i in range(len(suffixes)):
                suffix = suffixes[i]
                features['suffix:{}-gram={}'.format(len(suffix), suffix)] += 1

        if self.prev_token and word.idx > 0:
            features['token:prev={}'.format(sentence[word.idx-1].token)] += 1

        if self.next_token and word.idx < len(sentence) - 1:
            features['token:next={}'.format(sentence[word.idx+1].token)] += 1

        for wrd in sentence.get_left_window(word.idx, self.disjunctive_left_window):
            features['left:token={}'.format(wrd.token)] += 1

        for wrd in sentence.get_right_window(word.idx, self.disjunctive_right_window):
            features['right:token={}'.format(wrd.token)] += 1

        if self.tag_sequence_window > 0:
            features['tag:surrounding:sequence={}'.format('|'.join([
                wrd.tag for wrd in
                sentence.get_left_window(word.idx, self.tag_sequence_window) + [word] +
                sentence.get_right_window(word.idx, self.tag_sequence_window)
            ]))] += 1

        if named_entity is not None and named_entity.entity_gazette in self.gazetteer:
            features['in:clean:gazetteer'] = 1
            # features['gazette:{}'.format(named_entity.entity_gazette)] += 1

        if word.token in self.sloppy_gazetteer:
            features['in:sloppy:gazetteer'] += 1
            # for feature in self.sloppy_gazetteer[word.token]:
            # features['gazette:sloppy:{}'.format(feature)] += 1

        return features

    def get_instances_for_sentence(self, sentence, word_idx):
        """
        :param sentence: wikipedianer.corpus.base.Sentence
        :param word_idx: int
        """
        instances = []
        labels = []

        for unit in sentence.get_words_and_entities():
            if unit.name == "Word":
                if not self.valid_indices or word_idx in self.valid_indices:
                    instances.append(self._features_for_word(unit, sentence))
                    labels.append(unit.short_label)
                word_idx += 1
            else:
                for word in unit:
                    if not self.valid_indices or word_idx in self.valid_indices:
                        instances.append(self._features_for_word(word, sentence, unit))
                        labels.append(word.short_label)
                    word_idx += 1

        return instances, labels, word_idx


class WordVectorsExtractor(object):
    def __init__(self, model, window_size=5, valid_indices=None):
        self.model = model
        self.vector_size = self.model.vector_size
        self.window_size = window_size
        self.valid_indices = valid_indices if valid_indices is not None else set()

    @property
    def instance_vector_size(self):
        return self.window_size * self.vector_size

    def _vectors_for_word(self, word, sentence):
        word_window_vector = []

        full_word_window = sentence.get_left_window(word.idx, self.window_size) + [word] + \
            sentence.get_right_window(word.idx, self.window_size)

        for wrd in full_word_window:
            tokens = [s for s in wrd.tokens if s in self.model]

            if tokens:  # If there is an existing combination, take the best one (the first)
                word_window_vector.append(self.model[tokens[0]])
            else:  # If no possible combination is found, use a zero pad. TODO: What is the best solution?
                word_window_vector.append(np.zeros(self.vector_size, dtype=np.float32))

        window_vector = np.hstack(word_window_vector)  # Stack all the vectors in one large vector

        # Padding the window vector in case the predicate is located near the start or end of the sentence
        if word.idx - self.window_size < 0:  # Pad to left if the predicate is near to the start
            pad = abs(word.idx - self.window_size)
            window_vector = np.hstack((np.zeros(pad * self.vector_size, dtype=np.float32), window_vector))

        if word.idx + self.window_size + 1 > len(sentence):
            # Pad to right if the predicate is near to the end
            pad = word.idx + self.window_size + 1 - len(sentence)
            window_vector = np.hstack((window_vector, np.zeros(pad * self.vector_size, dtype=np.float32)))

        return window_vector

    def get_instances_for_sentence(self, sentence, word_idx):
        instances = []
        labels = []

        for word in sentence:
            if not self.valid_indices or word_idx in self.valid_indices:
                instances.append(self._vectors_for_word(word, sentence))
                labels.append(word.short_label)
            word_idx += 1

        return instances, labels, word_idx


class WikipediaCorpusColumnParser(object):
    def __init__(self, file_path, remove_stop_words=False,
                 keep_originals=False):
        self.file_path = file_path
        self.remove_stop_words = remove_stop_words
        self.keep_originals = keep_originals

    def __iter__(self):
        words = []
        has_named_entity = False
        original_line = ''

        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.decode('utf-8').strip()

                if line == "":
                    s = Sentence(words[:], has_named_entity)
                    words = []
                    has_named_entity = False
                    yield s
                else:
                    if self.keep_originals:
                        original_line = line
                    _, token, tag, class_string, head, dep = line.split()

                    widx = len(words)
                    is_doc_start = class_string.endswith('-DOC')

                    if not class_string.strip().startswith('O'):
                        has_named_entity = True
                        class_string = class_string.split('-DOC', 1)[0]
                        ner_tag, resources = class_string.split('-', 1)
                        wiki_uri, yago_uri, categories = resources.split('#', 3)
                        categories = categories.split('|')
                        wordnet_categories = [wc.split('-', 1)[0] for wc in categories
                                              if wc.split('-', 1)[0] in WORDNET_CATEGORIES_MOVIES or
                                              wc.split('-', 1)[0] in WORDNET_CATEGORIES_LEGAL]
                        yago_relations = [yr.split('-', 1)[0] for yr in categories
                                          if yr.split('-', 1)[0] in YAGO_RELATIONS_MOVIES]

                        words.append(Word(widx, token, tag, dep, head, ner_tag, yago_uri, wiki_uri,
                                          wordnet_categories, yago_relations, is_doc_start,
                                          original_string=original_line))
                    elif self.remove_stop_words and token in STOPWORDS_SET:
                        continue
                    else:
                        words.append(Word(widx, token, tag, dep, head, 'O',
                                          is_doc_start=is_doc_start,
                                          original_string=original_line))
