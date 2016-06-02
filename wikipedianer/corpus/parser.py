# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

from .base import Sentence, Word, WORDNET_CATEGORIES, YAGO_RELATIONS


class InstanceExtractor(object):
    def __init__(self, **kwargs):
        self.token = bool(kwargs.get('token', False))
        self.current_tag = bool(kwargs.get('current_tag', False))
        self.affixes = bool(kwargs.get('affixes', False))
        self.max_ngram_lenght = int(kwargs.get('max_ngram_length', 6))
        self.prev_token = bool(kwargs.get('prev_token', False))
        self.next_token = bool(kwargs.get('next_token', False))
        self.disjunctive_left_window = int(kwargs.get('disjunctive_left_window', 0))
        self.disjunctive_right_window = int(kwargs.get('disjunctive_right_window', 0))
        self.tag_sequence_window = int(kwargs.get('tag_sequence_window', 0))
        self.clean_gazetteer = kwargs.get('clean_gazette', {})
        self.sloppy_gazetteer = kwargs.get('sloppy_gazette', {})

    def _features_for_word(self, word, sentence, named_entity=None):
        """
        :param word: wikipedianer.corpus.base.Word
        :param sentence: wikipedianer.corpus.base.Sentence
        :param named_entity: wikipedianer.corpus.base.NamedEntity
        """
        features = {}

        if self.token:
            features['token:current'] = word.token

        if self.current_tag:
            features['tag:current'] = word.tag

        if self.affixes:
            prefixes, suffixes = word.get_affixes(self.max_ngram_lenght)

            for i in range(len(prefixes)):
                prefix = prefixes[i]
                features['prefix:{}-gram'.format(len(prefix))] = prefix

            for i in range(len(suffixes)):
                suffix = suffixes[i]
                features['suffix:{}-gram'.format(len(suffix))] = suffix

        if self.prev_token and word.idx > 0:
            features['token:prev'] = sentence[word.idx-1].token

        if self.next_token and word.idx < len(sentence) - 1:
            features['token:next'] = sentence[word.idx+1].token

        for wrd in sentence.get_left_window(word.idx, self.disjunctive_left_window):
            features['left:{}'.format(wrd.token)] = 1

        for wrd in sentence.get_right_window(word.idx, self.disjunctive_right_window):
            features['right:{}'.format(wrd.token)] = 1

        if self.tag_sequence_window > 0:
            features['tag:surrounding:sequence'] = ' '.join([
                wrd.tag for wrd in
                sentence.get_left_window(word.idx, self.tag_sequence_window) +
                sentence.get_right_window(word.idx, self.tag_sequence_window)
            ])

        if named_entity is not None and named_entity.entity_gazette in self.clean_gazetteer:
            for feature in self.clean_gazetteer[named_entity.entity_gazette]:
                features[feature] = 1

        if word.token in self.sloppy_gazetteer:
            for feature in self.sloppy_gazetteer[word.token]:
                features[feature] = 1

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


class WikipediaCorpusColumnParser(object):
    def __init__(self, file_path):
        self.file_path = file_path

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
                    else:
                        words.append(Word(widx, token, tag, dep, head, class_string))
