#!/usr/bin/env bash

function maxjobs {
  while [ `jobs | wc -l` -ge 5 ]
  do
    sleep 5
  done
}

find ../resources/docs_for_ner_conllx -type f -name "*.conll" | (while read file
do
  maxjobs
  filename=${file#../resources/docs_for_ner_conllx/}
  echo "Analyzing $filename"
  total_lines=$(grep "$filename" ../resources/conll_wc | awk '{ print $1 }')
  output_file=../resources/docs_for_ner_tagged/$filename
  ./tag_conllx_docs.py $file $output_file $total_lines &> logs/$filename &
done
wait)
