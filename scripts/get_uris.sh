#!/usr/bin/env bash

egrep -o '<a href="[^"]*">' wiki_* | egrep -o '"[^"]*"' | tr -d '"' | sort | uniq -c | sort -nr > uris_data.txt
