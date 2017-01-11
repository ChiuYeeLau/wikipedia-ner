# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, unicode_literals

import fnmatch
import os


CL_ITERATIONS = (
    'NER',
    'ENTITY',
    'LKIF',
    'YAGO',
    'URI'
)


def traverse_directory(path, file_pattern='*'):
    for root, _, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, file_pattern):
            yield os.path.join(root, filename)


