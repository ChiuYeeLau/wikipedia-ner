# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import cPickle
from collections import defaultdict
from nltk.corpus import stopwords
from .base import Sentence, Word, WORDNET_CATEGORIES, YAGO_RELATIONS


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
        self.clean_gazette = bool(kwargs.get('clean_gazette', False))

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

        if self.clean_gazette and named_entity is not None and named_entity.entity_gazette in self.gazetteer:
            features['gazette:clean:{}'.format(named_entity.entity_gazette)] += 1

        if word.token in self.sloppy_gazetteer:
            for feature in self.sloppy_gazetteer[word.token]:
                features['gazette:sloppy:{}'.format(feature)] += 1

        return features

    def get_instances_for_sentence(self, sentence):
        """
        :param sentence: wikipedianer.corpus.base.Sentence
        """
        instances = []
        labels = []

        for unit in sentence.get_words_and_entities():
            if unit.name == "Word":
                instances.append(self._features_for_word(unit, sentence))
                labels.append(unit.short_label)
            else:
                for word in unit:
                    instances.append(self._features_for_word(word, sentence, unit))
                    labels.append(word.short_label)

        return instances, labels


class FeatureExtractor(object):
    def __init__(self, **kwargs):
        self.features = defaultdict(int)
        self.max_ngram_length = kwargs.get('max_ngram_length', 0)
        self.disjunctive_left_window = int(kwargs.get('disjunctive_left_window', 0))
        self.disjunctive_right_window = int(kwargs.get('disjunctive_right_window', 0))
        self.tag_sequence_window = int(kwargs.get('tag_sequence_window', 0))

    def _features_from_word(self, word, sentence):
        self.features['in:clean:gazetteer'] += 1
        self.features['in:sloppy:gazetteer'] += 1

        self.features['token:current={}'.format(word.token)] += 1
        self.features['tag:current={}'.format(word.tag)] += 1

        prefixes, suffixes = word.get_affixes(self.max_ngram_length)

        for i in range(len(prefixes)):
            prefix = prefixes[i]
            self.features['prefix:{}-gram={}'.format(len(prefix), prefix)] += 1

        for i in range(len(suffixes)):
            suffix = suffixes[i]
            self.features['suffix:{}-gram={}'.format(len(suffix), suffix)] += 1

        if word.idx > 0:
            self.features['token:prev={}'.format(sentence[word.idx - 1].token)] += 1

        if word.idx < len(sentence) - 1:
            self.features['token:next={}'.format(sentence[word.idx + 1].token)] += 1

        for wrd in sentence.get_left_window(word.idx, self.disjunctive_left_window):
            self.features['left:token={}'.format(wrd.token)] += 1

        for wrd in sentence.get_right_window(word.idx, self.disjunctive_right_window):
            self.features['right:token={}'.format(wrd.token)] += 1

        if self.tag_sequence_window > 0:
            self.features['tag:surrounding:sequence={}'.format('|'.join([
                wrd.tag for wrd in
                sentence.get_left_window(word.idx, self.tag_sequence_window) + [word] +
                sentence.get_right_window(word.idx, self.tag_sequence_window)
            ]))] += 1

    def features_from_sentence(self, sentence):
        for unit in sentence.get_words_and_entities():
            if unit.name == "Word":
                self._features_from_word(unit, sentence)
            else:
                for word in unit:
                    self._features_from_word(word, sentence)

    def update_features(self, features):
        self.features.update(features)

    @property
    def sorted_features(self):
        return {feature: idx for idx, feature in enumerate(sorted(self.features))}

    def save_sorted_features(self, file_name):
        with open(file_name, 'wb') as f:
            cPickle.dump(self.sorted_features, f)

    def save_features(self, file_name):
        with open(file_name, 'wb') as f:
            cPickle.dump(self.features, f)


class WikipediaCorpusColumnParser(object):
    def __init__(self, file_path, remove_stop_words=False):
        self.file_path = file_path
        self.remove_stop_words = remove_stop_words

    def __iter__(self):
        words = []
        has_named_entity = False

        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.decode('utf-8').strip()

                if line == "":
                    s = Sentence(words[:], has_named_entity)
                    words = []
                    has_named_entity = False
                    yield s
                else:
                    _, token, tag, class_string, head, dep = line.split()

                    widx = len(words)

                    if class_string.strip() != 'O':
                        has_named_entity = True
                        ner_tag, resources = class_string.split('-', 1)
                        wiki_uri, yago_uri, categories = resources.split('#', 3)
                        categories = categories.split('|')
                        wordnet_categories = [wc.split('-', 1)[0] for wc in categories
                                              if wc.split('-', 1)[0] in WORDNET_CATEGORIES]
                        yago_relations = [yr.split('-', 1)[0] for yr in categories
                                          if yr.split('-', 1)[0] in YAGO_RELATIONS]

                        words.append(Word(widx, token, tag, dep, head, ner_tag, yago_uri, wiki_uri,
                                          wordnet_categories, yago_relations))
                    elif self.remove_stop_words and token in STOPWORDS_SET:
                        continue
                    else:
                        words.append(Word(widx, token, tag, dep, head, class_string))
