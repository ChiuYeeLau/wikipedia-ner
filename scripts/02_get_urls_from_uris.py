#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import requests
import sys
import urllib
from HTMLParser import HTMLParser
from urlparse import urlparse

WIKIPEDIA_BASENAME = u'en.wikipedia.org'
WIKIPEDIA_BASEPATH = u'/wiki/'
WIKIPEDIA_SCHEME = u'http'
WIKIPEDIA_NAMESPACE = '{}://{}{}'.format(WIKIPEDIA_SCHEME, WIKIPEDIA_BASENAME, WIKIPEDIA_BASEPATH)

incorrect_uris = set()
parsed_uris = set()
h = HTMLParser()

with open("parsed_uris.txt", "r") as pu:
    for l in pu.readlines():
        uri, _ = l.strip().split(',', 1)
        parsed_uris.add(uri)

with open("incorrect_uris.txt", "r") as iu:
    for l in iu.readlines():
        incorrect_uris.add(l.strip())

with open("parsed_uris.txt", "a") as pu:
    with open("incorrect_uris.txt", "a") as iu:
        with open("uris_data.txt", "r") as f:
            for l in f.readlines():
                try:
                    count, uri = l.strip().split(None, 1)
                except ValueError:
                    continue
                if uri.strip() in parsed_uris or uri.strip() in incorrect_uris or uri.strip().startswith("http")\
                    or "index.php" in uri or uri.strip().startswith("wikt"):
                    iu.write('{}\n'.format(uri))
                    continue
                if count < 10:
                    print >> sys.stderr, "Reached minimum count"
                    break
                print >> sys.stderr, "Getting URI for {}".format(uri)
                try:
                    unquoted_uri = urllib.unquote('{}{}'.format(WIKIPEDIA_NAMESPACE, uri)).decode('utf-8')
                    r = requests.get(h.unescape(unquoted_uri))
                    if r.status_code != 200:
                        iu.write('{}\n'.format(uri))
                        continue
                    parsed_url = urlparse(r.url)
                    if parsed_url.netloc != WIKIPEDIA_BASENAME:
                        iu.write('{}\n'.format(uri))
                        continue
                    final_url = u'{}://{}{}'.format(WIKIPEDIA_SCHEME, parsed_url.netloc, parsed_url.path)
                    pu.write('{},{}\n'.format(uri, final_url))
                    pu.flush()
                except Exception as e:
                    print >> sys.stderr, u'Exception for {}: {}'.format(uri, e)

print >> sys.stderr, "Finished parsing URIs"

