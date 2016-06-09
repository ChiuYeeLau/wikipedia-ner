#!/usr/bin/env bash


while [ -s ../resources/diff_ids.txt ]
do
    echo "Running python script"
    ./get_urls_from_ids.py
    echo
    echo "Running update"
    ./update_urls.sh
done
