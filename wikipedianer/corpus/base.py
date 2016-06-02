# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

WORDNET_CATEGORIES = {
    'wordnet_actor_109765278',
    'wordnet_film_director_110088200',
    'wordnet_film_maker_110088390',
    'wordnet_movie_106613686',
    'wordnet_soundtrack_104262969'
}

YAGO_RELATIONS = {
    'actedIn',
    'directed',
    'edited',
    'wroteMusicFor'
}


class Word(object):
    def __init__(self, token, tag, dep, head, ner_tag, yago_uri='', wiki_uri='',
                 wordnet_categories=None, yago_relations=None):
        self.token = token
        self.tag = tag.upper()
        self.dep = dep
        self.head = int(head)
        self.ner_tag = ner_tag
        self.yago_uri = yago_uri
        self.wiki_uri = wiki_uri
        self.wordnet_categories = wordnet_categories if wordnet_categories is not None else []
        self.yago_relations = yago_relations if yago_relations is not None else []

    def __to_string__(self):
        return '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t'.format(
            self.token,
            self.tag,
            self.head,
            self.dep,
            self.ner_tag,
            self.yago_uri,
            self.wiki_uri,
            '|'.join(self.wordnet_categories),
            '|'.join(self.yago_relations)
        ).strip()

    def __unicode__(self):
        return self.__to_string__()

    def __str__(self):
        return self.__to_string__().encode('utf-8')

    def __repr__(self):
        return str(self)

    @property
    def label(self):
        return '{}-{}:{}:{}:{}'.format(self.ner_tag, self.yago_uri, self.wiki_uri, '|'.join(self.wordnet_categories),
                                       '|'.join(self.yago_relations))

    @property
    def short_label(self):
        return '{}-{}'.format(self.ner_tag, self.yago_uri)

    @property
    def is_ner(self):
        return self.ner_tag != 'O'

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

    def tokens(self):
        """
        Method to get different combinations of tokens, with different use of capital letters
        in order to look for it in a embedding model.
        :return: A list of all possible combinations of a token or a lemma, ordered by importance
        """

        return [
            self.token,
            self.token.lower(),
            self.token.capitalize(),
            self.token.upper()
        ]


class Sentence(object):
    def __init__(self, words):
        self._words = words  # type: list[Word]

    def __to_string__(self):
        return '\n'.join(map(lambda (i, w): '{}\t{}'.format(i, w), enumerate(self, start=1)))

    def __iter__(self):
        for word in self._words:
            yield word

    def __getitem__(self, item):
        return self._words[item]

    def __unicode__(self):
        return self.__to_string__()

    def __str__(self):
        return self.__to_string__()

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self._words)

    def get_clean_gazettes(self):
        clean_gazettes = {}

        for entity in self.get_named_entities():
            entity_gazette = ' '.join([word.token for widx, word in entity])
            entity_feature = '{}:gazette:{}'.format(entity[0][1].yago_uri, len(entity))

            if entity_gazette not in clean_gazettes:
                clean_gazettes[entity_gazette] = set()

            clean_gazettes[entity_gazette].add(entity_feature)

        return clean_gazettes

    def get_sloppy_gazettes(self):
        sloppy_gazettes = {}

        for entity in self.get_named_entities():
            entity_feature = '{}:gazette:{}'.format(entity[0][1].yago_uri, len(entity))

            for widx, word in entity:
                if word not in sloppy_gazettes:
                    sloppy_gazettes[word] = set()

                sloppy_gazettes[word].add(entity_feature)

        return sloppy_gazettes

    def get_named_entities(self):
        named_entities = []

        for idx, word in [(i, w) for i, w in enumerate(self._words) if w.is_ner]:
            if word.is_ner_start:
                named_entities.append([])

            named_entities[-1].append((idx, word))

        return named_entities

    def get_left_window(self, loc, window_size):
        return self._words[max(0, loc - window_size):loc]

    def get_right_window(self, loc, window_size):
        return self._words[loc+1:loc+window_size+1]
