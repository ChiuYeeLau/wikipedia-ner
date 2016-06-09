#!/usr/bin/env bash

function maxjobs {
  while [ `jobs | wc -l` -ge 6 ]
  do
    sleep 5
  done
}

find ../resources/docs_for_ner_parsed -type f -name "*.conll" | (while read file
do
  maxjobs
  filename=${file#../resources/docs_for_ner_parsed/}
  echo "Analyzing $filename"
  total_lines=$(grep "$filename" ../resources/conll_wc_parsed | awk '{ print $1 }')
  ./merge_file.py $file ../resources/docs_for_ner_headers/$filename $total_lines > ../resources/docs_for_ner/$filename 2> logs/$filename &
done
wait)
