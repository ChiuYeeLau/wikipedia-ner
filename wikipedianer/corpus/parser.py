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
        self.clean_gazettes = kwargs.get('clean_gazette', {})
        self.sloppy_gazettes = kwargs.get('sloppy_gazette', {})

    def get_instances_for_sentence(self, sentence):
        """
        :param sentence: wikipedianer.corpus.base.Sentence
        """
        instances = []
        labels = []

        for entity in sentence.get_named_entities():
            entity_gazette = ' '.join([word.token for widx, word in entity])

            for widx, word in entity:
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

                if self.prev_token and widx > 0:
                    features['token:prev'] = sentence[widx-1].token

                if self.next_token and widx < len(sentence) - 1:
                    features['token:next'] = sentence[widx+1].token

                for wrd in sentence.get_left_window(widx, self.disjunctive_left_window):
                    features['left:{}'.format(wrd.token)] = 1

                for wrd in sentence.get_right_window(widx, self.disjunctive_right_window):
                    features['right:{}'.format(wrd.token)] = 1

                if self.tag_sequence_window > 0:
                    features['tag:surrounding:sequence'] = ' '.join([
                        wrd.tag for wrd in
                        sentence.get_left_window(widx, self.tag_sequence_window) +
                        sentence.get_right_window(widx, self.tag_sequence_window)
                    ])

                if entity_gazette in self.clean_gazettes:
                    for feature in self.clean_gazettes[entity_gazette]:
                        features[feature] = 1

                if word.token in self.sloppy_gazettes:
                    for feature in self.sloppy_gazettes[word.token]:
                        features[feature] = 1

                instances.append(features)
                labels.append(word.label)

        return instances, labels


class WikipediaCorpusColumnParser(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        words = []
        with open(self.file_path, 'r') as f:
            for line in f:
                line = line.decode('utf-8').strip()

                if line == "":
                    s = Sentence(words[:])
                    words = []
                    yield s
                else:
                    _, token, tag, class_string, head, dep = line.split()

                    if class_string.strip() != 'O':
                        ner_tag, resources = class_string.split('-', 1)
                        wiki_uri, yago_uri, categories = resources.split('#', 3)
                        categories = categories.split('|')
                        wordnet_categories = [wc.split('-', 1)[0] for wc in categories
                                              if wc.split('-', 1)[0] in WORDNET_CATEGORIES]
                        yago_relations = [yr.split('-', 1)[0] for yr in categories
                                          if yr.split('-', 1)[0] in YAGO_RELATIONS]

                        words.append(Word(token, tag, dep, head, ner_tag, yago_uri, wiki_uri,
                                          wordnet_categories, yago_relations))
                    else:
                        words.append(Word(token, tag, dep, head, class_string))
