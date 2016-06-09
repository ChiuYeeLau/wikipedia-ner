#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function

import sys
import urllib, json
from tqdm import tqdm

BASE_URL = "https://en.wikipedia.org/w/api.php?action=query&prop=info&inprop=url&format=json&pageids="
current_uids = []

with open("../resources/diff_ids.txt", "r") as f:
    uids = [l.strip() for l in f.readlines()]

for uid in tqdm(uids):
    if len(current_uids) < 50:
        current_uids.append(uid)
    else:
        query_url = BASE_URL + "|".join(current_uids)
        response = urllib.urlopen(query_url)
        data = json.loads(response.read())

        assert len(data["query"]["pages"]) == len(current_uids)

        with open("../resources/ids_urls.txt", "a") as f:
            for uid, pageinfo in data["query"]["pages"].iteritems():
                if "canonicalurl" in pageinfo:
                    f.write("{},{}\n".format(uid, pageinfo["canonicalurl"]).encode("utf-8"))

        current_uids = [uid]
