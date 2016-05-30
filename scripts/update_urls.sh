#!/usr/bin/env bash

cd ../resources
sort -n ./ids_urls.txt | uniq > /tmp/ids_urls.txt
mv /tmp/ids_urls.txt .
sed -i.bak 's/https/http/' ./ids_urls.txt
cut -d"," -f1 ./ids_urls.txt > ./parsed_ids.txt
diff ./parsed_ids.txt ./docs_ids.txt | grep ">" | awk '{ print $2 }' > ./diff_ids.txt
