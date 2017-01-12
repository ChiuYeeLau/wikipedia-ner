# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

from collections import defaultdict

WORDNET_CATEGORIES_MOVIES = {
    'wordnet_actor_109765278',
    'wordnet_film_director_110088200',
    'wordnet_film_maker_110088390',
    'wordnet_movie_106613686',
    'wordnet_soundtrack_104262969'
}

WORDNET_CATEGORIES_LEGAL = {
    'wordnet_adjudicator_109769636',
    'wordnet_court_108329453',
    'wordnet_criminal_record_106490173',
    'wordnet_judge_110225219',
    'wordnet_judiciary_108166318',
    'wordnet_jurisdiction_108590369',
    'wordnet_law_100611143',
    'wordnet_law_106532330',
    'wordnet_law_108441203',
    'wordnet_lawyer_110249950',
    'wordnet_legal_code_106667792',
    'wordnet_legal_document_106479665',
    'wordnet_legal_power_105198427',
    'wordnet_party_110402824',
    'wordnet_pleading_106559365'
}

YAGO_RELATIONS_MOVIES = {
    'actedIn',
    'directed',
    'edited',
    'wroteMusicFor'
}


class Word(object):
    name = "Word"

    def __init__(self, idx, token, tag, ner_tag, uri_label='', yago_labels=list(), lkif_labels=list(),
                 entity_labels=list(), is_doc_start=False, original_string=''):
        self.idx = int(idx)
        self.token = token
        self.tag = tag.upper()
        self.ner_tag = ner_tag
        self.uri_label = uri_label
        self.yago_labels = yago_labels
        self.lkif_labels = lkif_labels
        self.entity_labels = entity_labels
        self.is_doc_start = is_doc_start
        self.original_string = original_string

    def __to_string__(self):
        return '{}\t{}\t{}\t{}\t{}\t{}'.format(
            self.token,
            self.tag,
            self.uri_label,
            '|'.join(self.yago_labels),
            '|'.join(self.lkif_labels),
            '|'.join(self.entity_labels)
        ).strip()

    def __str__(self):
        return self.__to_string__()

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.original_string == other.original_string

    @property
    def labels(self):
        if self.is_ner:
            return ('I-%s' % self.uri_label, self.yago_labels,
                    self.lkif_labels, self.entity_labels, 'I')
        else:
            return 'O', ['O'], ['O'], ['O'], 'O'

    @property
    def io_label(self):
        if self.is_ner:
            return 'I-{}'.format(self.uri_label)
        else:
            return 'O'

    @property
    def is_ner(self):
        return not self.ner_tag == 'O'

    @property
    def is_ner_start(self):
        return self.ner_tag == 'B'

    def get_affixes(self, max_ngram_length=6):
        prefixes = []
        suffixes = []

        for i in range(1, min(len(self.token)-1, max_ngram_length+1)):
            prefixes.append(self.token[:i])
            suffixes.append(self.token[-i:])

        return prefixes, suffixes

    @property
    def tokens(self):
        """
        Method to get different combinations of tokens, with different use of capital letters
        in order to look for it in a embedding model.
        :return: A list of all possible combinations of a token or a lemma, ordered by importance
        """

        return (
            self.token,
            self.token.lower(),
            # FIXME: Is it better to add the following?
            # self.token.capitalize(),
            # self.token.upper()
        )


class NamedEntity(object):
    name = "NamedEntity"

    def __init__(self, words):
        self._words = words

    def __iter__(self):
        for word in self._words:
            yield word

    def __repr__(self):
        return ' '.join([word.token for word in self._words])

    @property
    def entity_gazette(self):
        return ' '.join([word.token for word in self._words])


class Sentence(object):
    def __init__(self, words, has_named_entity=False):
        self._words = words  # type: list[Word]
        self.has_named_entity = has_named_entity

    def __to_string__(self):
        return '\n'.join(map(lambda w: '%s\t%s' % (w[0], w[1]), enumerate(self, start=1)))

    def __iter__(self):
        for word in self._words:
            yield word

    def __getitem__(self, item):
        return self._words[item]

    def __str__(self):
        return self.__to_string__()

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._words)

    def __eq__(self, other):
        if len(self._words) != len(other._words):
            return False
        for word1, word2 in zip(self._words, other._words):
            if not word1 == word2:
                return False
        return True

    @property
    def labels(self):
        return [word.io_label for word in self]

    def get_gazettes(self):
        gazettes = defaultdict(int)

        for entity in self.get_named_entities():
            gazettes[' '.join([word.token for word in entity])] += 1

        return gazettes

    def get_named_entities(self):
        named_entities = []

        for word in [w for w in self._words if w.is_ner]:
            if word.is_ner_start:
                named_entities.append([])
            
            if len(named_entities) == 0:
                named_entities.append([word])
            else:
                named_entities[-1].append(word)

        return named_entities

    def get_unique_properties(self, property_name):
        """Returns a set of values of property_name in all words in sentence"""
        result = set()
        for word in self._words:
            if not hasattr(word, property_name):
                continue
            target = getattr(word, property_name)
            if isinstance(target, list):
                if len(target) > 0:
                    result.add(target[0])
            else:
                result.add(target)
        return result

    def get_words_and_entities(self):
        named_entity = []

        for word in self._words:
            if not word.is_ner:
                if len(named_entity) > 0:
                    yield NamedEntity(named_entity)
                named_entity = []
                yield word
            elif word.is_ner_start:
                if len(named_entity) > 0:
                    yield NamedEntity(named_entity)
                named_entity = [word]
            else:
                named_entity.append(word)

        if len(named_entity) > 0:
            yield NamedEntity(named_entity)

    def get_left_window(self, loc, window_size):
        return self._words[max(0, loc - window_size):loc]

    def get_right_window(self, loc, window_size):
        return self._words[loc+1:loc+window_size+1]

    def get_original_strings(self):
        """Returns a list with the original_lines of every word, in order."""
        return [word.original_string for word in self._words]
